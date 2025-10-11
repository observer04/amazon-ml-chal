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
