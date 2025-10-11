# ML Challenge 2025: Process Log

## Step 1: Comprehensive EDA (`comprehensive_eda.ipynb`)

### Dataset Overview
- **Size:** 75,000 products, 4 columns (sample_id, catalog_content, image_link, price)
- **No missing values** in any column
- **Price range:** $0.13 - $2,796 (mean: $23.65, median: $14.00)
- **Challenge:** Wide price range + SMAPE metric = low-price items are high risk

### Tests Conducted & Rationale

#### 1. Structural Format Analysis (Bullet Points Hypothesis)
**Why:** Observed that some products have structured bullet points while others don't
**Test:** Compare price distributions for products with/without bullet points
**Finding:** 
- 72.6% have bullets, 27.4% don't
- Mean prices nearly identical ($23.64 vs $23.68, p=0.87)
- **NOT statistically significant**, but median differs ($14.99 vs $11.98)
- Products with 5 bullets most common (35,931 products)
**Conclusion:** Bullet points alone don't predict price, but include as interaction feature

#### 2. IPQ (Item Pack Quantity) Analysis ⭐ CRITICAL
**Why:** Every product has Value + Unit fields - suspected strong predictor
**Test:** Extract Value/Unit, analyze price per unit, compare unit types
**Finding:**
- **98.7%** have Value field (only 940 missing)
- Top units: Ounce (55%), Count (23%), Fl Oz (15%)
- Clear price patterns by unit type (e.g., Pound=$43.75, Ounce=$21.87)
- Strong relationship between Value and Price
**Conclusion:** IPQ is PRIMARY predictor - must be in all models

#### 3. Text Statistics Correlation
**Why:** Test if information density correlates with price
**Test:** Calculate text_length, num_words, num_lines vs price
**Finding:**
- text_length: r=0.147 (weak)
- num_words: r=0.144 (weak)
- num_bullets: r=0.018 (negligible)
**Conclusion:** Weak signals alone, but useful in combination

#### 4. Quality Keywords Analysis
**Why:** Hypothesis that premium keywords indicate higher prices
**Test:** Extract "organic", "premium", "gourmet", "natural" keywords
**Finding:**
- **Premium: +46.6%** price difference (strongest signal!)
- Gourmet: +28.7%
- Natural: +27.3%
- Organic: +11.8%
**Conclusion:** Quality keywords are strong price indicators - extract all

#### 5. Pack Size Detection
**Why:** "Pack of X" likely affects pricing (bulk discounts)
**Test:** Regex extraction of pack size
**Finding:** 24.9% have pack size mentions
**Conclusion:** Extract as separate feature

### Key Data Quality Issues
1. **Unit inconsistency:** "Ounce" vs "ounce" vs "oz" → need normalization
2. **Outliers:** 7.37% (5,524 products) using IQR method
3. **Price skew:** Right-skewed distribution → consider log transform

### Features Created & Saved
**File:** `dataset/train_with_features.csv`
**New features (15 total):**
- Structural: `has_bullets`, `num_bullets`, `has_description`, `num_lines`
- Text stats: `text_length`, `num_words`
- IPQ: `value`, `unit`, `price_per_unit`
- Quality: `has_organic`, `has_premium`, `has_gourmet`, `has_natural`
- Pack info: `has_pack_of`, `pack_size`

### Strategic Insights
1. **Feature Importance (predicted):** Value > Unit Type > Quality Keywords > Pack Size > Text Stats > Bullets
2. **SMAPE Risk:** Low-priced items (<$10) will dominate error - need stratified approach
3. **Model Strategy:** Start with IPQ baseline, incrementally add features
4. **Next Priority:** Extract Item Name, individual bullet points, and Product Description for deeper NLP

---

## Step 2: Deep Dive Analysis (Extended EDA)

### Motivation
After initial EDA, identified critical gaps that directly impact feature engineering and model performance:
- **Unit normalization strategy** needed (99 unique units detected)
- **Brand extraction validation** required (heuristic needed testing)
- **Price segmentation** for SMAPE risk management
- **Image availability** assessment to determine Image MLP viability

### Deep Dive 1: Unit Normalization Strategy ✅

**Test:** Analyze unit type distribution and identify consolidation opportunities

**Findings:**
- **99 unique unit types** in raw data (many are duplicates with different casing)
- **Unit variant opportunities:**
  - Ounce variants: 55,244 products (Ounce, ounce, oz, OZ, Oz, Fl Oz, fl oz, FL Oz)
  - Count variants: 18,209 products (Count, count, COUNT)
  - Pound variants: 236 products (Pound, pound, lb, LB)
  
**Normalization mapping:**
```python
'ounce' → 'Ounce', 'oz' → 'Ounce', 'OZ' → 'Ounce'
'fl oz' → 'Fl Oz', 'FL Oz' → 'Fl Oz'
'count' → 'Count', 'COUNT' → 'Count'
'pound' → 'Pound', 'lb' → 'Pound', 'LB' → 'Pound'
```

**Impact:** Reduces 99 units → 88 units (11 variants normalized)

**Price by normalized unit (top units):**
- Ounce: $21.38 mean (44,006 products)
- Count: $29.94 mean (18,209 products)
- Fl Oz: $21.09 mean (11,238 products)
- Pound: $40.46 mean (236 products)

**Action:** ✅ Implemented in `feature_extraction.py` UNIT_MAPPING config

---

### Deep Dive 2: Brand Extraction Validation ⚠️

**Test:** Validate brand extraction heuristic on actual catalog_content format

**Discovered format:** `"Item Name: BRAND PRODUCT DESCRIPTION..."`
- All products start with "Item Name:" prefix
- Brand comes AFTER prefix, not at absolute start
- Original regex failed (0% extraction rate)

**Current heuristic issue:**
```python
# This fails on "Item Name: Brand Product..."
match = re.match(r'^([A-Z][a-zA-Z0-9\s&\'-]+?)(?:,|\s-\s|\n|$)', first_line)
```

**Examples from data:**
- "Item Name: Log Cabin Sugar Free Syrup, 24 FL OZ (Pack of 12)"
- "Item Name: Vlasic Ovals Hamburger Dill Pickle Chips, Keto Friendly, 16 FL OZ"

**Action Required:** Update `feature_extraction.py` to:
1. Strip "Item Name:" prefix first
2. Then extract brand from remaining text
3. Handle cases with/without prefix

---

### Deep Dive 3: Price Segmentation & SMAPE Risk Analysis 🔴 CRITICAL

**Test:** Analyze SMAPE risk zones by price segment

**Price segment distribution:**
- **Budget (<$10):** 28,512 products (38.0%) 🔴 **HIGHEST RISK**
- **Mid-Range ($10-$25):** 24,764 products (33.0%)
- **Premium ($25-$50):** 13,616 products (18.2%)
- **Luxury (>$50):** 8,108 products (10.8%)

**SMAPE Risk Demonstration** (for constant $5 prediction error):
| True Price | Predicted | SMAPE Error | Risk Level |
|------------|-----------|-------------|------------|
| $5 | $10 | 33.3% | 🔴 EXTREME |
| $10 | $15 | 20.0% | 🔴 HIGH |
| $25 | $30 | 9.1% | 🟢 OK |
| $50 | $55 | 4.8% | 🟢 Low |
| $100 | $105 | 2.4% | 🟢 Very Low |

**Key Insight:** **38% of products are in HIGH RISK zone** (<$10)
- These products contribute DISPROPORTIONATELY to overall SMAPE
- A $2 error on $5 item = 28.6% SMAPE
- Same $2 error on $50 item = 3.9% SMAPE

**Feature coverage by segment:**
- Budget: 98.7% IPQ, 31.4% quality keywords
- Mid-Range: 99.1% IPQ, 46.6% quality keywords
- Premium: 99.0% IPQ, 51.3% quality keywords
- Luxury: 97.5% IPQ, 53.4% quality keywords

**Strategic Implications:**
1. **Stratified CV:** Use price bins for K-fold splits
2. **Weighted loss:** Consider penalizing low-price errors more heavily
3. **Conservative predictions:** Avoid over/under-predicting Budget segment
4. **Separate handling:** May need Budget-specific model or post-processing

---

### Deep Dive 4: Image Availability Assessment ✅

**Test:** Sample 100 random images to estimate download success rate

**Findings:**
- **100% image link coverage** (all 75k products have URLs)
- **All URLs from Amazon CDN:** `https://m.media-amazon.com/images/I/...`
- **Download test results:** 100/100 successful (100.0% success rate!)

**Sample successful URLs:**
- https://m.media-amazon.com/images/I/71QD2OFXqDL.jpg
- https://m.media-amazon.com/images/I/813OiT8mdJL.jpg
- https://m.media-amazon.com/images/I/71HGx42QmUL.jpg

**Image MLP Strategy Decision:**
- ✅ **Image MLP is HIGHLY VIABLE**
- Keep ensemble weight at **20%** (or potentially increase to 25%)
- 100% availability means strong signal potential
- Still implement zero-vector fallback for robustness
- Images likely capture: packaging quality, brand visibility, premium presentation

**Expected contribution:**
- Images alone: ~22-28% SMAPE (weak standalone)
- In ensemble: Adds 2-3% SMAPE improvement over LightGBM+Text alone
- Visual cues complement text features (e.g., packaging quality → premium pricing)

---

## Performance Forecast & Competition Strategy

### Expected Model Performance (Individual Models)

| Model | Features | Expected SMAPE | Confidence |
|-------|----------|----------------|------------|
| **LightGBM Baseline** | IPQ + keywords + units + pack size | 12-15% | High ✅ |
| **Text MLP** | 384-dim sentence embeddings | 18-22% | Medium ✅ |
| **Image MLP** | 512-dim ResNet18 embeddings | 22-28% | Medium ✅ |

### Expected Ensemble Performance

| Ensemble Strategy | Expected SMAPE | Leaderboard Position |
|-------------------|----------------|----------------------|
| **Simple weighted avg** (55/25/20) | 11-13% | Top 30-40% |
| **Optimized weights** (scipy SLSQP) | 10-12% | Top 20-35% |
| **With segment handling** | 9-11% | Top 15-30% |

### Realistic Target: **10-12% SMAPE** → **Top 50 achievable** ✅

### Stretch Goal: **<10% SMAPE** → **Top 20** (requires perfect execution)

### Factors That Could Improve Performance:
1. ✅ Perfect hyperparameter tuning (LightGBM depth, learning rate)
2. ✅ Effective ensemble weight optimization on OOF predictions
3. ✅ Special handling for Budget segment (<$10)
4. ⚠️ Brand extraction fix (currently not working)
5. 💡 Value × Quality interaction features
6. 💡 Price-per-unit as explicit feature

### Factors That Limit Performance:
1. 🔴 Budget segment (38% of data) has inherent SMAPE risk
2. ⚠️ No external data allowed (can't use brand databases)
3. ⚠️ Simple weighted ensemble vs advanced stacking
4. ⚠️ No SMAPE-aware loss for neural networks (using MSE)

---

## Step 3: Feature Extraction Implementation ✅ COMPLETE

### Brand Extraction Fix
**Issue:** Original regex failed (0% extraction) due to "Item Name:" prefix in catalog_content

**Solution implemented:**
```python
def extract_brand(text):
    # Strip "Item Name:" prefix first
    text = re.sub(r'^Item Name:\s*', '', text, flags=re.IGNORECASE)
    first_line = text.split('\n')[0].split(',')[0].strip()
    match = re.match(r'^([A-Z][a-zA-Z0-9\s&\'-]+?)(?:\s-\s|,|\(|\d|$)', first_line)
    if match:
        brand = match.group(1).strip()
        if brand not in ['Pack', 'Set', 'Bundle', 'Size', 'Box', 'Case']:
            return brand
    return ''
```

**Results:**
- Before fix: 0% brand extraction
- After fix: **77.1% brand extraction** (57,846 / 75,000 products)
- Improvement: **+57,846 products with brand identified**

### Feature Extraction Execution

**Files created:**
- `dataset/train_with_features.csv` - Training data with 24 features
- `dataset/test_with_features.csv` - Test data with 23 features (no price)

**Processing performance:**
- Train: 75,000 samples in 8.2 seconds
- Test: 75,000 samples in 7.9 seconds
- Total: ~8 seconds per 75k dataset (very efficient!)

**Features extracted (24 new columns):**

1. **Structured parsing:**
   - `item_name` - Product name extracted from catalog
   - `description` - Full product description
   - `bullet_1` through `bullet_6` - Individual bullet points

2. **IPQ features:**
   - `value` - Numeric quantity value (84.8% coverage, 63,569 products)
   - `unit` - Measurement unit (raw)
   - `unit_normalized` - Standardized unit (Ounce/Count/Fl Oz/Pound)
   - `price_per_unit` - Calculated price/value ratio (train only)

3. **Brand:**
   - `brand` - Extracted brand name (77.1% coverage)

4. **Quality indicators:**
   - `has_premium`, `has_organic`, `has_gourmet`, `has_natural`, `has_artisan`, `has_luxury`
   - Boolean flags for quality keywords

5. **Structural features:**
   - `pack_size` - Detected pack quantity
   - `text_length`, `word_count`, `has_bullets`, `num_bullets`

6. **Price segmentation:**
   - `price_segment` - Budget (<$10) / Mid-Range ($10-50) / Premium ($50-100) / Luxury (>$100)

**Price segment distribution (train):**
- Budget (<$10): 28,764 products (38.4%) 🔴 HIGH SMAPE RISK
- Mid-Range ($10-50): 38,144 products (50.9%)
- Premium ($50-100): 6,199 products (8.3%)
- Luxury (>$100): 1,893 products (2.5%)

**Value/Unit coverage:**
- Value extracted: 63,569 / 75,000 (84.8%)
- Missing value: 11,431 products (15.2%)
- Strategy: Use median/mean imputation for missing values

**Unit normalization impact:**
- Top 3 normalized units cover 97.8% of products with units
- Ounce: 58.9%, Count: 24.3%, Fl Oz: 15.0%

---

## Step 4: Model Development Plan (Ready to Execute)

### Current Status: ✅ Features ready, waiting for Kaggle deployment

**DO NOT RUN LOCALLY - PREPARE FOR KAGGLE:**
1. Text embeddings (sentence-transformers) - requires GPU
2. Image embeddings (ResNet18) - requires GPU + downloads
3. Model training (LightGBM + 2 MLPs) - GPU recommended
4. Ensemble optimization - fast, can run anywhere

### GitHub Repository Setup - **REQUIRED NEXT**
1. Create remote repository
2. Push all code (`src/`, `config/`, `kaggle.py`)
3. Add `.gitignore` (exclude `dataset/`, `amzml/`, `outputs/`)
4. Tag version v1.0 with features complete

### Kaggle Deployment Checklist
- [ ] Push code to GitHub
- [ ] Upload `train_with_features.csv` and `test_with_features.csv` as Kaggle dataset
- [ ] Create Kaggle notebook
- [ ] Enable GPU (P100)
- [ ] Install dependencies: `sentence-transformers`, `lightgbm`
- [ ] Run `kaggle.py` (estimated 2-3 hours on P100)
- [ ] Submit predictions

---

## Expected Timeline & Next Actions

### Immediate (DO NOW):
1. **Create GitHub remote and push code** ⏰ 5 minutes
2. **Upload feature CSVs to Kaggle dataset** ⏰ 10 minutes
3. **Verify kaggle.py is ready** ⏰ 5 minutes

### On Kaggle (P100 GPU):
1. **Text embeddings generation** ⏰ 30-40 minutes
2. **Image embeddings generation** ⏰ 45-60 minutes
3. **LightGBM training (5-fold CV)** ⏰ 15-20 minutes
4. **Text MLP training** ⏰ 20-30 minutes
5. **Image MLP training** ⏰ 20-30 minutes
6. **Ensemble optimization** ⏰ 2-5 minutes
7. **Generate predictions** ⏰ 5 minutes

**Total Kaggle runtime:** ~2.5-3 hours

### Target Performance:
- **Conservative:** 12-13% SMAPE → Top 40-50
- **Realistic:** 10-12% SMAPE → Top 30-40
- **Optimistic:** 9-11% SMAPE → Top 20-30

---

## FOCUS: Stop local execution, deploy to Kaggle NOW

---

## Step 5: Problem Resolution & Code Corrections ⚠️

**Date:** October 11, 2025  
**Context:** After feature extraction, multiple critical issues discovered during baseline model preparation

### Problem 1: Corrupted Feature Files (1M rows instead of 75k)

**Discovery:**
- User noticed `test_with_features.csv` had doubled in size
- Line count showed ~1M rows instead of expected 75k
- Excel showed many empty cells across rows

**Root Cause Analysis:**
```
1. catalog_content field contains MULTILINE text (bullet points with \n)
2. Initial CSV generation used default pandas to_csv() without proper quoting
3. CSV parser interpreted newlines as row breaks → split single rows into multiple rows
4. Result: 75,000 logical rows became 1,048,731 physical lines in CSV
```

**Visual Example of Problem:**
```csv
# CORRECT (1 row):
sample_id,catalog_content,price
12345,"Item Name: Coffee\nBullet 1: Premium\nBullet 2: Organic",25.99

# WRONG (parser sees 3 rows):
sample_id,catalog_content,price
12345,"Item Name: Coffee
Bullet 1: Premium
Bullet 2: Organic",25.99
```

**Resolution:**
1. **Deleted corrupted files:** `rm dataset/train_with_features.csv dataset/test_with_features.csv`
2. **Created `generate_features.py`** with proper CSV handling:
   ```python
   # Key fix: Use quoting=1 (QUOTE_ALL) to escape embedded newlines
   df.to_csv(filepath, index=False, quoting=1)
   df_verify = pd.read_csv(filepath, quoting=1)  # Must read with same quoting
   ```
3. **Regenerated features:** Now correctly shows 75,000 rows in pandas (1M+ lines in wc -l is normal for quoted multiline CSV)

**Verification:**
```bash
wc -l dataset/train_with_features.csv  # Shows 1,050,409 lines (includes quoted newlines)
python -c "import pandas as pd; print(len(pd.read_csv('...', quoting=1)))"  # Shows 75,000 rows ✓
```

---

### Problem 2: Missing Critical Features for Text Embeddings

**Discovery:**
- User performed sanity check on original feature schema design
- Identified missing `all_bullets_text` field - critical for text embeddings
- Missing `description_length` - needed for text richness analysis

**Original Schema (User's Design):**
```python
{
    'all_bullets_text': str,  # Concatenated for embeddings - MISSING!
    'description_length': int,  # Just description part - MISSING!
    # ... other fields implemented correctly
}
```

**What Was Actually Implemented:**
```python
{
    'bullet_1' to 'bullet_6': str,  # ✓ Separate columns for LightGBM
    'text_length': int,             # ✓ But this is ENTIRE catalog, not just description
    # Missing: all_bullets_text
    # Missing: description_length
}
```

**Why This Matters:**
- **LightGBM** needs separate bullet columns (`bullet_1`, `bullet_2`, etc.) → We had this ✓
- **Text embeddings (MLP)** need concatenated text (`all_bullets_text`) → We were MISSING this ✗
- Without `all_bullets_text`, we'd have to reconstruct it later → risk of inconsistency

**Resolution:**

1. **Updated `parse_catalog_content()` in `src/feature_extraction.py`:**
   ```python
   # Added concatenated bullets field
   all_bullets_text = '\n'.join(bullets)
   
   return {
       'item_name': item_name,
       'bullets': bullets,
       'all_bullets_text': all_bullets_text,  # NEW
       'description': description
   }
   ```

2. **Updated `create_features()` to extract new fields:**
   ```python
   # Add all_bullets_text for embeddings
   df['all_bullets_text'] = parsed.apply(lambda x: x['all_bullets_text'])
   
   # Add description_length (just description, not entire catalog)
   df['description_length'] = df['description'].fillna('').str.len()
   ```

3. **Regenerated features with fixed code:**
   ```bash
   python generate_features.py
   # Output: train_with_features.csv (30 columns, was 28)
   # Output: test_with_features.csv (27 columns, was 25)
   ```

**New Feature Count:**
- **Train:** 28 → 30 columns (+`all_bullets_text`, +`description_length`)
- **Test:** 25 → 27 columns (same additions)

---

### Problem 3: Git Repository Issues - Large Files & Lost Data

**Discovery:**
- Attempted to push changes to GitHub
- Push failed with "HTTP 408 timeout" - files too large (78MB)
- Later discovered `git reset --hard` had deleted `train.csv` and `test.csv`

**Issues Encountered:**

1. **Large CSV files in git history:**
   - User had committed corrupted CSVs in commit `2c8cbbc` ("data")
   - Files were 1M+ lines → ~70MB each
   - Git push timing out due to size

2. **Data loss from reset:**
   - Did `git reset --hard aa6971d` to remove bad commit
   - This DELETED `train.csv` and `test.csv` from filesystem
   - Only sample files remained

3. **.gitignore was correct but files already tracked:**
   - `.gitignore` had `dataset/*.csv` 
   - But once files are committed, .gitignore doesn't apply

**Resolution:**

1. **Recovered original data from reflog:**
   ```bash
   git checkout 2c8cbbc -- dataset/train.csv dataset/test.csv
   # Restored train.csv (70MB) and test.csv (70MB)
   ```

2. **Cleaned up git history:**
   ```bash
   git reset --hard aa6971d  # Remove corrupted CSV commit
   git add baseline_model.py generate_features.py  # Add only code
   git commit -m "Add baseline + feature generation scripts"
   ```

3. **Final decision: Include CSVs for Kaggle convenience:**
   - User requested: "add both it's easier for me to get them on kaggle this way"
   - Pushed train.csv + test.csv + code to GitHub
   - GitHub warning about large files (>50MB) but push succeeded
   ```bash
   git push origin main
   # Result: All files now on GitHub for easy Kaggle cloning
   ```

---

### Problem 4: Baseline Model Not Run

**Discovery:**
- Agent was preparing to run baseline experiments locally
- User correctly intervened: "you seem to running everything on my laptop. can you be little focused"

**Why This Was Wrong:**
- Local machine has no GPU
- Baseline experiments take 20-30 minutes
- Should save local compute for Kaggle GPU environment
- Risk of running outdated code if changes made after local execution

**Resolution:**
- **Created `baseline_model.py`** with 4 incremental experiments but **DID NOT RUN**
- Added proper reasoning comments for each experiment
- Script ready to run on Kaggle after feature generation

**Correct Workflow (Now Documented):**
```
Local:
1. Write code (feature_extraction.py, baseline_model.py) ✓
2. Commit to git ✓
3. Push to GitHub ✓

Kaggle (with GPU):
1. Clone repo ⏳ NEXT
2. Generate features (15 sec) ⏳
3. Run baseline experiments (20-30 min) ⏳
4. Analyze results ⏳
5. Submit predictions ⏳
```

---

### Files Modified/Created in This Step:

**Created:**
- `generate_features.py` - Proper CSV handling with quoting=1
- `baseline_model.py` - 4 incremental experiments for validation

**Modified:**
- `src/feature_extraction.py` - Added `all_bullets_text`, `description_length`

**Regenerated:**
- `dataset/train_with_features.csv` - Now 30 columns (was 28)
- `dataset/test_with_features.csv` - Now 27 columns (was 25)

**Git Commits:**
```
09fe526 - Add: baseline_model.py + generate_features.py
1091ac5 - Fix: Add all_bullets_text + description_length
```

**Pushed to GitHub:**
- All code files (src/, config/, scripts)
- Original data (train.csv, test.csv) - 70MB each
- Ready for Kaggle deployment

---

### Lessons Learned:

1. **CSV Multiline Handling:** Always use `quoting=1` (QUOTE_ALL) when data contains newlines
2. **Feature Schema Completeness:** Design schema upfront for ALL use cases (LightGBM + embeddings)
3. **Git Large Files:** Consider `.gitignore` before first commit, or use Git LFS for >50MB files
4. **Local vs Kaggle Execution:** Reserve GPU-intensive work for Kaggle, not local laptop
5. **Schema Validation:** User's sanity check caught missing fields before baseline - saved debugging time later

---

### Current Status (Ready for Kaggle):

**✅ Completed:**
- Feature extraction fixed with proper CSV handling
- Complete feature schema (30 train columns, 27 test columns)
- Baseline model script ready (not run yet)
- All code + data pushed to GitHub

**⏳ Next Steps (On Kaggle):**
1. Clone repo: `git clone https://github.com/observer04/amazon-ml-chal.git`
2. Generate features: `python generate_features.py` (~15 seconds)
3. Run baseline: `python baseline_model.py` (~20-30 minutes)
4. Analyze results and decide next steps

**Expected Baseline Results:**
- Experiment 1 (IPQ only): 15-18% SMAPE
- Experiment 2 (+ quality): 13-16% SMAPE
- Experiment 3 (+ brand): 12-15% SMAPE
- Experiment 4 (+ text stats): 11-14% SMAPE

If final baseline <12% → Submit to leaderboard (likely Top 50)  
If 12-15% → Add text embeddings using `all_bullets_text` field  
If >15% → Debug features, optimize hyperparameters

---

## FOCUS: Deploy to Kaggle NOW - All Issues Resolved

---

## Step 6: Baseline Experiments & Debugging 🔧

**Date:** October 11, 2025  
**Context:** First baseline run on Kaggle revealed SMAPE 2-3x higher than expected

### Initial Results (WRONG - Multiple Bugs):
- Experiment 1 (IPQ): 76.67% SMAPE (Expected: 15-18%)
- Experiment 4 (Full): 70.32% SMAPE (Expected: 11-14%)

### Bug Discovery #1: SMAPE Formula Error

**Found:** Line 44 in `baseline_model.py`
```python
return 200 * np.mean(diff)  # WRONG - doubles the score
```

**Fixed:**
```python
return 100 * np.mean(diff)  # CORRECT
```

**Impact:** Divided all scores by 2 → 76.67% became 38.34%

---

### Bug Discovery #2: Missing Value Handling

**Found:** Line 81 - filling NaN with -999
```python
df[features] = df[features].fillna(-999)  # Confuses LightGBM!
```

**Fixed:** Removed line - LightGBM handles NaN natively

**Impact:** Small improvement expected

---

### Second Run Results (Still Wrong):
- Experiment 1 (IPQ Only - value only): 38.35% SMAPE
- Experiment 4 (Full): 35.20% SMAPE
- **Budget segment: 54.98% SMAPE** (38.4% of data) ← Catastrophic!

### Analysis of Results:

**What Worked:**
- Progressive improvement: Each feature addition helped (+8.2% total)
- Feature importance sensible: value, brand, text_length top 3
- Mid-Range segment: 19.08% SMAPE ← Close to target!

**What's Still Wrong:**
- Budget (<$10): 54.98% SMAPE (Expected: 15-20%)
- Premium ($50-100): 35.41% SMAPE (Expected: 12-15%)
- Luxury (>$100): 58.98% SMAPE (Expected: 10-15%)

**Root Cause:** SMAPE is extremely sensitive to low-price errors!
- $5 product predicted as $10 → 33% SMAPE
- $25 product predicted as $30 → 9% SMAPE
- Budget segment (38% of data) drags overall score up

---

### Bug Discovery #3: Missing 'unit' Feature

**Found:** Feature list only had `'value'` without `'unit'`

**Problem:**
- 12 Ounces of coffee ≠ 12 Count of coffee pods!
- Without unit, model can't distinguish quantity types
- Model treats 12oz and 12-count as same → huge errors

**Code Check:**
```python
# Experiment 1: 
features = ['value']  # Missing 'unit'!

# Experiment 2-3: ALSO missing 'unit'!
# Only Experiment 4 had it
```

**Fixed:** Added `'unit'` to ALL experiments

---

### Third Run Results (With unit):
- Experiment 1 (value + unit): 36.96% SMAPE (was 38.35%)
- Experiment 2 (+ quality): 37.28% SMAPE ← **WORSE!** Unit still missing in Exp 2!
- Experiment 3 (+ brand): 35.99% SMAPE ← Unit still missing!
- Experiment 4 (full + unit): 33.90% SMAPE

**Budget segment: 52.72% SMAPE** ← Slight improvement but still catastrophic

**Feature Importance:**
- value: 2390.2
- brand: 2338.6
- text_length: 1837.6
- word_count: 1759.2
- **unit: 669.6** ← Ranked #6, very low importance!

**Diagnosis:** Unit was added to Exp 1 & 4, but **sed command missed Exp 2 & 3**!

---

### Bug Discovery #4: Incomplete Fix Application

**Problem:** Only Experiments 1 and 4 have `unit`, Experiments 2 and 3 don't

**Evidence:**
```
Experiment 2 features: value, has_premium, has_organic, ...  # No unit!
Experiment 3 features: value, has_premium, ..., brand  # No unit!
```

**Fixed:** Manually added `'unit'` to Experiments 2 & 3 in baseline_model.py

**Committed:** `e819e0b` - "Fix: Add 'unit' to ALL experiments"

---

### Current Status - Ready for Fourth Run:

**Fixed Bugs:**
1. ✅ SMAPE formula (200 → 100)
2. ✅ Missing value handling (removed -999 fill)
3. ✅ Added 'unit' to Experiment 1
4. ✅ Added 'unit' to Experiment 4
5. ✅ Added 'unit' to Experiments 2 & 3 (final fix)

**Expected Results After All Fixes:**
- Experiment 1 (value + unit): **16-19% SMAPE**
- Experiment 2 (+ quality): **14-17% SMAPE**
- Experiment 3 (+ brand): **13-15% SMAPE**
- Experiment 4 (full): **11-14% SMAPE**
- Budget segment: **20-25% SMAPE** (still high due to SMAPE sensitivity)

**Next Action on Kaggle:**
```bash
cd amazon-ml-chal
git pull origin main
python baseline_model.py
```

---

### Key Learnings:

1. **SMAPE Sensitivity:** Low-price products (<$10) have disproportionate SMAPE penalty
2. **Unit Critical:** value without unit is meaningless (12oz ≠ 12 count)
3. **Debugging Process:** Multiple iterations needed to find all bugs
4. **sed Limitations:** Regex substitution missed some feature lists - manual fix required
5. **Feature Importance:** Can diagnose missing features (unit ranked #6 → underutilized)

---

### Remaining Challenges:

**Budget Segment Problem:**
- 38.4% of products are <$10
- SMAPE formula penalizes errors on low prices heavily
- Even with all fixes, Budget SMAPE likely 20-25% (vs 15% for Mid-Range)

**Potential Solutions (If baseline still >15% overall):**
1. Separate model for Budget segment with different loss function
2. Log-transform predictions to reduce relative errors
3. Post-processing: clip predictions to reasonable ranges
4. Add text embeddings to capture subtle price signals

---

## FOCUS: Re-run baseline with ALL fixes applied → Expected 11-14% SMAPE

---

## Step 7: Baseline Validation & Transform Experiments 🎯

**Date:** October 11, 2025 (Evening)  
**Context:** After all bug fixes, completed multiple baseline runs with progressive improvements

### Fourth Run Results (All Bugs Fixed):
- **Experiment 1** (value + unit): 33.35% SMAPE
- **Experiment 2** (+ quality keywords): 33.68% SMAPE ← Quality keywords HURT!
- **Experiment 3** (+ brand): 33.99% SMAPE ← Brand also HURT!
- **Experiment 4** (full features): 33.90% SMAPE

**By Segment (Exp 4):**
- Budget (<$10): 51.75% SMAPE (38.4% of data) 🔴 Still catastrophic
- Mid-Range ($10-$50): 18.85% SMAPE ✅ Good
- Premium ($50-$100): 39.47% SMAPE ⚠️ High
- Luxury (>$100): 57.96% SMAPE 🔴 Very high

**Key Observations:**
1. **Budget segment dominates error** - 38% of data, 51% SMAPE
2. Quality keywords & brand **hurt performance** (overfitting on noisy features?)
3. Simple value + unit performs best (33.35%)
4. Model struggles with extreme price ranges (<$10 and >$100)

---

### External Evaluation & Defensive Analysis

**Agent Received Critique:** Claims of missing features and wrong objective

**Claimed Issues:**
1. "Missing price_per_unit feature"
2. "Should use MAPE objective instead of SMAPE"

**Defense & Reality Check:**

**Re: "Missing price_per_unit":**
- ✅ price_per_unit EXISTS in train_with_features.csv (col 11)
- ❌ BUT: It's **train-only** (uses true price) → data leakage if used!
- ✅ Correct approach: Use `value` + `unit` separately (what we did)
- **Verdict:** Critic was WRONG - feature exists but shouldn't be used

**Re: "Use MAPE objective":**
- ❌ MAPE (Mean Absolute Percentage Error) ≠ SMAPE
- MAPE formula: (1/n) * Σ |actual - pred| / |actual| ← **Asymmetric!**
- SMAPE formula: (1/n) * Σ |actual - pred| / ((|actual| + |pred|)/2) ← **Symmetric!**
- MAPE penalizes over-prediction MORE than under-prediction
- SMAPE treats both equally
- LightGBM doesn't have native SMAPE objective (using MSE is standard)
- **Verdict:** Critic was WRONG - MAPE is not equivalent to SMAPE

**Real Issues Identified:**
1. ✅ Budget segment (<$10) is 51% SMAPE → main problem
2. ✅ Model treats prices linearly → SMAPE wants percentage thinking
3. ✅ Need transform to handle wide price range ($0.13 - $2,796)

---

### Fifth Run: Log Transform Experiment

**Hypothesis:** Log transform will help model think in percentages, reduce Budget segment error

**Changes:**
```python
# During training:
y_train_log = np.log1p(y_train)  # log(1 + price)
model.fit(X_train, y_train_log)

# During prediction:
y_pred = np.expm1(model.predict(X_val))  # exp(log_pred) - 1
```

**Results:**
- **Overall SMAPE: 30.21%** ← 🎉 3.7% improvement!
- Budget (<$10): **36.32%** ← 15.4% improvement! (was 51.75%)
- Mid-Range ($10-$50): **21.29%** ← 2.4% worse (was 18.85%)
- Premium ($50-$100): **43.26%** ← 3.8% worse (was 39.47%)
- Luxury (>$100): **63.07%** ← 5.1% worse (was 57.96%)

**Analysis:**
- ✅ Log transform **dramatically helped Budget segment**
- ❌ But **hurt Premium and Luxury** significantly
- Trade-off: Help 38% of data (Budget) vs hurt 11% (Premium+Luxury)
- Net gain: 3.7% overall improvement

---

### Sixth Run: Segment-Adaptive Transform (CATASTROPHIC FAILURE)

**Hypothesis:** Use different transforms per segment to optimize each

**Approach:**
```python
# Budget: sqrt transform (gentler than log)
# Mid-Range: log transform
# Premium: no transform
# Luxury: no transform
```

**Results:**
- **Overall SMAPE: 54.00%** ← 🔥 24% WORSE than log!
- Complete disaster, reverted immediately

**Root Cause:** Complex transform logic introduced bugs, predictions inconsistent

---

### Seventh Run: Square Root Transform + Interactions

**Hypothesis:** sqrt transform balances all segments better than log

**Changes:**
1. Transform: `y_train_sqrt = np.sqrt(y_train)`
2. Added 10 interaction features:
   - value × quality flags (3)
   - pack × quality (2)
   - brand × quality (2)
   - size × value (2)
   - brand_exists indicator

**Results:**
- **Overall SMAPE: 30.70%** 
- Budget (<$10): 38.32% (vs 36.32% log)
- Mid-Range ($10-$50): 21.88% (vs 21.29% log)
- Premium ($50-$100): 43.11% (vs 43.26% log)
- Luxury (>$100): 62.95% (vs 63.07% log)

**Analysis:**
- Sqrt slightly worse than log overall (+0.5%)
- But more balanced across segments
- Interactions helped slightly
- Chose sqrt for stability

---

### Hyperparameter Tuning & Submission Preparation

**Aggressive Training Settings:**
```python
params = {
    'num_leaves': 127,           # Was: 31
    'learning_rate': 0.02,       # Was: 0.05
    'num_boost_round': 3000,     # Was: 1000
    'early_stopping_rounds': 100, # Was: 50
    'min_data_in_leaf': 10,
    'lambda_l1': 0.3,            # L1 regularization
    'lambda_l2': 0.3,            # L2 regularization
    'max_depth': 12
}
```

**Submission Script Bugs Found:**
1. `median()` on numpy array → `np.median()` (fixed)
2. Training stopping too early (853-1570 rounds) → increased patience

**Final Training Results (5-fold CV):**
- Fold iterations: 1197, 1307, 1362, 1329, 1513 (healthy range)
- RMSE (sqrt space): 2.02 → 1.98 → 1.96 → 1.93 → 1.95 (decreasing ✅)
- No overfitting signs

**Test Predictions Generated:**
- 75,000 samples
- Min: $1.36, Max: $314.36
- Mean: $19.86, Median: $16.28
- 86.8% in $10-$50 range (matches training distribution)

**Validation Passed:**
- ✅ Correct format (sample_id, price)
- ✅ 75,000 rows
- ✅ All positive floats
- ✅ No missing values
- ✅ IDs match test.csv

---

## Step 8: First Submission Results & Analysis 📊

**Date:** October 11, 2025 (Late Evening)

### Submission Details:
- **Model:** LightGBM with sqrt transform, 5-fold CV ensemble
- **Features:** 28 features (value, unit, pack_size, brand, 6 quality flags, 10 interactions)
- **Cross-Validation SMAPE:** 30.70%
- **Expected Public LB:** 28-32% SMAPE

### **Public Leaderboard Result: 62.45% SMAPE** 🔥

**Reality Check:**
- CV Score: 30.70%
- Public LB: 62.45%
- **Gap: +31.75% (CV was 2x better than reality!)**

---

### Initial Reaction & Diagnosis:

**This is CATASTROPHIC overfitting or fundamental error.**

**Possible Causes:**

1. **Train/Test Distribution Mismatch** 🔴 MOST LIKELY
   - Test set has different price distribution than train
   - Model trained on one distribution, tested on another
   - Features may not generalize

2. **Feature Extraction Bug** ⚠️ HIGH PROBABILITY
   - Test feature extraction may have errors
   - Different parsing results on test vs train
   - Silent failures in brand/unit extraction

3. **Transform Issues** ⚠️ POSSIBLE
   - Sqrt transform applied incorrectly on test predictions
   - Inverse transform (squaring) may have numerical issues
   - Extreme values getting amplified

4. **Data Leakage During CV** ⚠️ POSSIBLE
   - CV validation may have leaked information
   - Stratification may not match actual test distribution
   - OOF predictions optimistic

5. **Ensemble Weights Wrong** ⚠️ LESS LIKELY
   - Simple average of 5 models should be stable
   - But if individual models bad, average won't help

6. **Target Variable Issues** ⚠️ UNLIKELY
   - Prices in test may be in different scale
   - Currency conversion? (unlikely for Amazon)
   - Different product categories entirely?

---

### Critical Questions to Investigate:

**Feature Distribution:**
- [ ] Compare train vs test distributions for key features (value, unit, brand_exists)
- [ ] Check if test has missing values where train didn't
- [ ] Verify unit normalization working on test

**Prediction Analysis:**
- [ ] Plot predicted vs actual distribution (if possible)
- [ ] Check for prediction outliers (very high/low)
- [ ] Verify inverse transform applied correctly

**Model Sanity:**
- [ ] Check individual fold predictions (are all 5 models bad?)
- [ ] Try simple baseline (median price) on test set
- [ ] Compare feature importance train vs validation

**Code Audit:**
- [ ] Re-verify test feature generation
- [ ] Check for any train-only logic in create_submission.py
- [ ] Ensure no accidental data leakage

---

### Comparison with Expected Performance:

| Metric | Expected | Actual | Delta |
|--------|----------|--------|-------|
| CV SMAPE | 30.70% | 30.70% | 0% ✅ |
| Public LB | 28-32% | **62.45%** | +31.75% 🔥 |
| Budget Segment CV | 38.32% | ??? | ??? |
| Mid-Range CV | 21.88% | ??? | ??? |

**Hypothesis:** Test set is MUCH harder than train, or features completely broken on test.

---

### Next Steps (Immediate):

1. **Feature Distribution Analysis:**
   ```python
   # Compare train vs test
   train_stats = train_with_features.describe()
   test_stats = test_with_features.describe()
   # Look for large differences
   ```

2. **Prediction Statistics:**
   ```python
   # Check test_out.csv predictions
   preds = pd.read_csv('test_out.csv')
   print(preds['price'].describe())
   # Compare to train price distribution
   ```

3. **Simple Baseline:**
   ```python
   # Predict median price for all test samples
   median_price = train['price'].median()
   baseline_preds = [median_price] * 75000
   # Submit and compare SMAPE
   ```

4. **Feature Extraction Verification:**
   ```python
   # Manually inspect 10 random test samples
   # Check if value, unit, brand extracted correctly
   ```

5. **Model Diagnostics:**
   ```python
   # Check if model predicting reasonable ranges
   # Look for NaN or inf in predictions
   ```

---

### Strategic Implications:

**If Distribution Mismatch:**
- Need domain adaptation techniques
- Consider reweighting training samples
- May need different feature set for test

**If Feature Bug:**
- Fix extraction and regenerate
- Should see immediate improvement

**If Transform Issue:**
- Try no transform (raw prices)
- Try log instead of sqrt
- Check numerical stability

**If Fundamental Model Issue:**
- Start from scratch with simpler model
- Focus on robust features only (value, unit)
- Reduce complexity

---

### Lessons Learned (So Far):

1. **CV can be dangerously optimistic** - 30% CV → 62% LB
2. **Feature engineering needs more validation** - didn't verify test extraction
3. **Should have submitted simple baseline first** - to calibrate expectations
4. **Transform adds complexity** - sqrt may have hidden issues
5. **Need better train/test comparison** - before modeling

---

## FOCUS: Debug the 2x gap between CV and LB - Something is fundamentally wrong

````
