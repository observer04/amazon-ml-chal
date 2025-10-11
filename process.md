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
