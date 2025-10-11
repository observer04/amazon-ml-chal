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
