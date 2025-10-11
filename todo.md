# TODO: Road to 30-40% SMAPE 🎯

**Target:** 30-40% SMAPE on Public Leaderboard  
**Current Status:** 62.45% SMAPE (First submission - text-only baseline)  
**Resources:** Unlimited Kaggle GPU time (P100 or 2xT4)  
**Key Insight:** Images are critical (50% of solution based on post-mortem)

---

## 🚨 CRITICAL RULES (Don't Break These!)

1. ✅ **Use BOTH text AND images** - Competition explicitly mentioned this
2. ✅ **Submit incrementally** - Test each component separately to isolate value
3. ✅ **No sample_test_out.csv reliance** - Treat it as unreliable (user confirmed)
4. ✅ **Validate on Public LB** - Each submission teaches us about test distribution
5. ✅ **Build safety nets** - Ensemble should degrade gracefully if components fail

---

## 📊 DECISION TREE: Strategic Options

```
ROOT: How to reach 30-40% SMAPE?
│
├─[OPTION A] 🎯 RECOMMENDED: Incremental Multi-Modal Build
│  ├─ Step 1: Baseline + Simple LightGBM (validate foundation)
│  ├─ Step 2: Add Text Embeddings (test text value)
│  ├─ Step 3: Add Image Embeddings (test image value)
│  └─ Step 4: Optimize Ensemble (fine-tune weights)
│  
├─[OPTION B] 🚀 AGGRESSIVE: Full Multi-Modal Ensemble (teammates' approach)
│  ├─ Build all 3 branches in parallel
│  ├─ Ensemble: 30% LightGBM + 20% Text + 50% Image
│  └─ Single submission, higher risk/reward
│
├─[OPTION C] 🧪 EXPERIMENTAL: Alternative Architectures
│  ├─ Vision Transformer (ViT) instead of ResNet
│  ├─ CLIP for joint text-image embeddings
│  └─ Multi-modal fusion transformers
│
└─[OPTION D] 🔬 CONSERVATIVE: Feature Engineering Deep Dive
   ├─ Extract more text features (brand, specs, etc.)
   ├─ Unit normalization and quantity detection
   └─ Stay with LightGBM, avoid deep learning
```

---

## 🎯 OPTION A: Incremental Multi-Modal Build (RECOMMENDED)

**Philosophy:** "Learn what works by testing each component"  
**Time:** 3-4 submissions over 2-3 days  
**Risk:** Low (each step validated)  
**Expected Outcome:** 32-38% SMAPE

### A.1: Baseline + Robust LightGBM

**Goal:** Establish reliable foundation, beat 62% baseline

#### Tasks:
- [ ] **A.1.1** Create simple baseline model (median by unit)
  - Extract IPQ (value, unit) from catalog_content
  - Calculate median price per unit type (Ounce, Count, Fl Oz)
  - Predict test using unit medians
  - **Expected SMAPE:** 50-55%
  - **Purpose:** Know the "do nothing smart" baseline

- [ ] **A.1.2** Build LightGBM with ONLY robust features
  - Features to use (100% coverage, never fail):
    - `value` (numerical, from IPQ)
    - `unit` (categorical, from IPQ - one-hot encode)
    - `text_length` (length of catalog_content)
    - `word_count` (word count)
    - `num_bullets` (bullet point count)
  - Features to AVOID (may fail on test):
    - `brand` (77% coverage - might be lower on test)
    - `has_premium`, `has_organic`, etc. (keyword flags - brittle)
    - `pack_size` (25% coverage - too sparse)
  - Train 5-fold CV
  - **Expected CV:** 38-42%
  - **Expected LB:** 42-48%
  - **Purpose:** Robust baseline that won't catastrophically fail

- [ ] **A.1.3** Submit LightGBM-only prediction
  - Generate `test_out_lgb_only.csv`
  - Submit to leaderboard
  - **Decision point:** 
    - If LB <48%: ✅ Foundation is solid, proceed to A.2
    - If LB >48%: ⚠️ Debug features, check distribution mismatch

**Estimated Time:** 2-3 hours (feature extraction + training)

---

### A.2: Add Text Embeddings

**Goal:** Test if semantic text understanding helps

#### Tasks:
- [ ] **A.2.1** Generate text embeddings
  - Model: `sentence-transformers/all-MiniLM-L6-v2` (384-dim, fast)
  - Alternative: `sentence-transformers/all-mpnet-base-v2` (768-dim, better quality)
  - Input: Full `catalog_content` text (don't parse, use raw)
  - Output: `train_text_embeddings.npy` (75k × 384)
  - **Time:** 30-40 min on P100

- [ ] **A.2.2** Train Text MLP
  - Architecture: 384 → 256 → 128 → 1
  - Activation: ReLU
  - Dropout: 0.2
  - Loss: MSE (or log-transform target)
  - Epochs: 50-100 with early stopping
  - **Expected CV:** 45-55% (standalone)
  - **Time:** 20-30 min

- [ ] **A.2.3** Ensemble LightGBM + Text MLP
  - Simple average: 60% LightGBM + 40% Text MLP
  - Alternative: Optimize weights on CV folds
  - **Expected LB:** 38-44%
  - **Decision point:**
    - If LB improved: ✅ Text helps, proceed to A.3
    - If LB same/worse: ⚠️ Text not adding value, skip to images

**Estimated Time:** 1-1.5 hours

---

### A.3: Add Image Embeddings (THE BIG ONE)

**Goal:** Add visual features (packaging, size, quality)

#### Tasks:
- [ ] **A.3.1** Download all images
  - Use provided `src/utils.py::download_images()`
  - Train: 75k images → `images/train/`
  - Test: 75k images → `images/test/`
  - **Expected success rate:** 95%+ (all Amazon CDN)
  - **Time:** 1.5-2 hours (with 100 worker threads)
  - **Disk space:** ~15-20 GB total

- [ ] **A.3.2** Generate image embeddings
  - **Option 1 (Recommended):** ResNet18 (fast, 512-dim)
    - Pretrained: `torchvision.models.resnet18(pretrained=True)`
    - Extract from penultimate layer (before classifier)
    - **Time:** 45-60 min on P100
  - **Option 2:** ResNet50 (slower, 2048-dim)
    - Better quality, 3x slower
    - **Time:** 2-2.5 hours on P100
  - **Option 3:** EfficientNet-B0 (efficient, 1280-dim)
    - Best speed/quality tradeoff
    - **Time:** 1-1.5 hours on P100
  - Handle failures: Zero vector for missing/corrupted images
  - Output: `train_image_embeddings.npy`, `test_image_embeddings.npy`

- [ ] **A.3.3** Train Image MLP
  - Architecture: 512 → 256 → 128 → 1 (for ResNet18)
  - Same training as Text MLP
  - **Expected CV:** 45-55% (standalone)
  - **Time:** 20-30 min

- [ ] **A.3.4** Three-Model Ensemble
  - **Option 1:** Equal weights (33/33/33)
  - **Option 2:** Teammates' suggestion (30% LGB + 20% Text + 50% Image)
  - **Option 3:** Optimize weights using scipy on CV predictions
  - **Expected LB:** 32-38% 🎯
  - **Decision point:**
    - If LB 32-38%: ✅ SUCCESS! Move to A.4 for optimization
    - If LB 38-45%: 🟡 Decent, but optimize ensemble
    - If LB >45%: ⚠️ Debug image embeddings or model training

**Estimated Time:** 4-5 hours total

---

### A.4: Optimize Ensemble

**Goal:** Squeeze last 2-5% improvement

#### Tasks:
- [ ] **A.4.1** Grid search ensemble weights
  - Try combinations: 
    - 40/20/40 (balanced)
    - 25/25/50 (image-heavy)
    - 35/15/50 (teammates' approach)
    - 30/20/50 (slight image focus)
  - Evaluate on CV folds
  - **Expected gain:** 1-3% SMAPE

- [ ] **A.4.2** Add baseline model to ensemble
  - Include simple median-by-unit heuristic (10-15% weight)
  - Acts as safety net
  - Final: 15% Baseline + 25% LGB + 20% Text + 40% Image
  - **Expected gain:** 1-2% SMAPE (reduces extreme errors)

- [ ] **A.4.3** Prediction clipping and post-processing
  - Clip predictions to [0.50, 500] (train range with buffer)
  - Try log-averaging instead of linear averaging
  - Blend with training median for stability
  - **Expected gain:** 0.5-2% SMAPE

- [ ] **A.4.4** Try different text/image models
  - Text: Swap to `all-mpnet-base-v2` (better quality)
  - Image: Try EfficientNet or ViT
  - **Expected gain:** 2-4% SMAPE if models better

**Estimated Time:** 2-3 hours

---

## 🚀 OPTION B: Aggressive Full Multi-Modal (Teammates' Approach)

**Philosophy:** "Build everything at once, submit once"  
**Time:** 1 submission, 6-8 hours prep  
**Risk:** High (if fails, don't know which component is broken)  
**Expected Outcome:** 30-40% SMAPE (if all works) or 45-55% (if issues)

### B.1: Parallel Development

#### Tasks:
- [ ] **B.1.1** Branch 1: Text Features → LightGBM (30% weight)
  - Extract: value, unit, pack_size, brand, quality keywords
  - Include interactions: value × quality, pack × brand
  - Train LightGBM with aggressive hyperparameters
  - **Expected CV:** 35-42%

- [ ] **B.1.2** Branch 2: Text Embeddings → MLP (20% weight)
  - Sentence-BERT: `all-mpnet-base-v2` (higher quality)
  - MLP: 768 → 512 → 256 → 128 → 1
  - Train with log-transformed targets
  - **Expected CV:** 42-50%

- [ ] **B.1.3** Branch 3: Image Embeddings → MLP (50% weight)
  - ResNet50 or EfficientNet-B0
  - MLP: 2048 → 512 → 256 → 128 → 1
  - Handle missing images: zero vectors
  - **Expected CV:** 40-48%

- [ ] **B.1.4** Weighted Ensemble (30-20-50 split)
  - Combine all three models
  - Clip predictions to reasonable range
  - **Expected LB:** 30-42% 🎯

**Pros:**
- ✅ Faster to first "good" submission
- ✅ If works, immediately competitive
- ✅ Follows teammates' proven architecture concept

**Cons:**
- ❌ Can't isolate which component helps/hurts
- ❌ If fails, hard to debug (3 models × features × embeddings)
- ❌ Higher risk of wasted compute if architecture wrong

**Estimated Time:** 6-8 hours (all parallel)

---

## 🧪 OPTION C: Experimental Architectures

**Philosophy:** "Try cutting-edge approaches for potential breakthrough"  
**Time:** Multiple submissions, high experimentation  
**Risk:** Very High (unproven for this problem)  
**Expected Outcome:** 25-40% SMAPE (huge variance)

### C.1: Vision Transformer (ViT)

#### Tasks:
- [ ] **C.1.1** Replace ResNet with ViT-B/16
  - Model: `google/vit-base-patch16-224`
  - Better for fine-grained visual details
  - Extract 768-dim embeddings
  - **Time:** 2-3 hours for embedding generation
  - **Expected improvement over ResNet:** 2-5% SMAPE (if ViT better for packaging)

**Pros:**
- ✅ State-of-art vision model
- ✅ Better attention to details (logos, text on packaging)

**Cons:**
- ❌ Slower (3x vs ResNet18)
- ❌ Unproven benefit for e-commerce images
- ❌ Requires more memory (may need 2xT4 instead of P100)

---

### C.2: CLIP (Joint Text-Image)

#### Tasks:
- [ ] **C.2.1** Use OpenAI CLIP for joint embeddings
  - Model: `openai/clip-vit-base-patch32`
  - Encode text and images in same space (512-dim)
  - Train MLP on concatenated [text_emb, image_emb]
  - **Expected benefit:** Captures text-image alignment (e.g., "premium" text + gold packaging)

**Pros:**
- ✅ Leverages text-image relationships
- ✅ Pre-trained on product-like images

**Cons:**
- ❌ Complex to implement correctly
- ❌ May not help if text/image correlation weak
- ❌ Slower than separate models

---

### C.3: Multi-Modal Fusion Transformer

#### Tasks:
- [ ] **C.3.1** Build attention-based fusion
  - Concat text + image embeddings
  - Add transformer layers for cross-attention
  - Learn optimal feature weighting
  - **Expected benefit:** 3-8% if fusion helps, or 0% if overkill

**Pros:**
- ✅ Theoretically optimal for multi-modal data
- ✅ Can learn complex interactions

**Cons:**
- ❌ Very complex to implement and tune
- ❌ High risk of overfitting
- ❌ Much slower training
- ❌ May not help given tabular nature of some features

---

## 🔬 OPTION D: Conservative Feature Engineering

**Philosophy:** "Squeeze everything from features, avoid deep learning complexity"  
**Time:** 2-3 days of iterative feature engineering  
**Risk:** Medium (limited upside without images)  
**Expected Outcome:** 38-45% SMAPE (unlikely to reach 30-40% without images)

### D.1: Advanced Text Feature Extraction

#### Tasks:
- [ ] **D.1.1** Unit normalization
  - Standardize units: "oz" → "Ounce", "lb" → "Pound", "fl oz" → "Fl Oz"
  - Reduce 99 unique units → 15-20 standard units
  - **Expected gain:** 2-3% SMAPE

- [ ] **D.1.2** Brand embeddings
  - Extract brand names (77% coverage)
  - Calculate average price per brand
  - Create brand frequency features
  - **Expected gain:** 1-2% SMAPE

- [ ] **D.1.3** Quantity detection
  - Regex: "Pack of X", "X Count", "Set of X"
  - Normalize to standard quantities
  - **Expected gain:** 1-2% SMAPE

- [ ] **D.1.4** Product category inference
  - Keyword-based: "Coffee", "Chocolate", "Vitamin", "Snack"
  - Category-specific price distributions
  - **Expected gain:** 2-4% SMAPE

- [ ] **D.1.5** Feature interactions
  - value × has_premium
  - value × pack_size
  - unit × brand
  - text_length × value
  - **Expected gain:** 1-3% SMAPE

**Total Expected:** 42-48% SMAPE with LightGBM only

**Pros:**
- ✅ No dependency on images (if download fails)
- ✅ Fast iteration (no GPU needed for embeddings)
- ✅ Interpretable features

**Cons:**
- ❌ Unlikely to reach 30-40% target without images
- ❌ Diminishing returns on feature engineering
- ❌ Ignores competition hints about images

---

## 🎯 MY RECOMMENDATION: Choose Your Path

### If you want **MAXIMUM CONFIDENCE** in reaching 30-40%:
→ **Choose OPTION A** (Incremental Multi-Modal)
- You learn what works at each step
- Can pivot if something doesn't help
- Lower risk, proven approach
- **4 submissions, 3-4 days**

### If you want **FASTEST PATH** to 30-40%:
→ **Choose OPTION B** (Aggressive Full Build)
- One big submission with all components
- Higher risk but faster if it works
- Follows teammates' architecture
- **1 submission, 1-2 days**

### If you want **POTENTIAL BREAKTHROUGH** to <30%:
→ **Choose OPTION C** (Experimental)
- Try ViT or CLIP for better image understanding
- Higher variance, higher potential upside
- Requires more experimentation
- **Multiple submissions, 4-5 days**

### If you want **SAFETY** (no image dependency):
→ **Choose OPTION D** (Feature Engineering)
- Unlikely to reach 30-40% but solid 40-45%
- No risk of image download failures
- Fast iteration
- **2-3 submissions, 2-3 days**

---

## 📋 PRE-FLIGHT CHECKLIST (Do Before ANY Submission)

- [ ] ✅ All required files exist (train.csv, test.csv)
- [ ] ✅ Feature extraction runs without errors
- [ ] ✅ Model training completes (no OOM, no crashes)
- [ ] ✅ Predictions generated for ALL 75k test samples
- [ ] ✅ Output CSV format matches sample_test_out.csv (sample_id, price)
- [ ] ✅ All predictions are positive floats (no NaN, no negative)
- [ ] ✅ Prediction statistics look reasonable:
  - Min: >$0.10 (not zero)
  - Max: <$1000 (not extreme)
  - Mean: $15-$40 (ballpark of train)
  - Median: $10-$30 (ballpark of train)
- [ ] ✅ File saved as `test_out.csv` or similar
- [ ] ✅ Git commit with clear message (e.g., "Add: Image embeddings ensemble")

---

## 🎮 DECISION MATRIX: Which Option Fits You?

| Criteria | Option A | Option B | Option C | Option D |
|----------|----------|----------|----------|----------|
| **Reach 30-40% SMAPE** | 90% | 80% | 60% | 30% |
| **Time to first submit** | 3h | 8h | 10h+ | 4h |
| **Total time needed** | 12-15h | 8-10h | 20-30h | 10-12h |
| **Risk level** | Low | Medium | High | Low |
| **Learning value** | High | Medium | High | Medium |
| **Debuggability** | High | Low | Medium | High |
| **Follows teammates** | Partial | Yes | No | No |
| **Uses images** | Yes | Yes | Yes | No |
| **Code complexity** | Medium | Medium | High | Low |
| **Kaggle GPU usage** | 12-15h | 8-10h | 20-30h | 2-3h |

---

## 🚀 RECOMMENDED EXECUTION PLAN

### My Suggestion: **OPTION A** with fallback to **OPTION B** if time pressure

**Week 1 (Days 1-3):**
- Day 1: A.1 - Baseline + LightGBM-only → Submit
  - **Target:** <48% LB (beat 62% baseline)
- Day 2: A.2 - Add Text Embeddings → Submit
  - **Target:** <44% LB
- Day 3: A.3 - Add Image Embeddings → Submit
  - **Target:** <38% LB 🎯

**If after Day 3 you're at 38-40% LB:**
- ✅ You're in target range!
- Spend Day 4-5 on A.4 (optimization) to get 32-36%

**If after Day 3 you're at 42-48% LB:**
- 🟡 Close but not there yet
- Try Option C.1 (ViT) or C.2 (CLIP) for breakthrough
- OR deep dive into A.4 optimization

**If after Day 3 you're still >48% LB:**
- ⚠️ Something is wrong
- Debug: Check train/test feature distributions
- Fallback: Try Option B from scratch with different models

---

## 📝 NEXT IMMEDIATE ACTIONS (Tell Me What You Want)

Before I write any code, tell me:

1. **Which option do you want to pursue?**
   - [ ] Option A (Incremental) - RECOMMENDED
   - [ ] Option B (Aggressive) - Teammates' approach
   - [ ] Option C (Experimental) - ViT/CLIP
   - [ ] Option D (Conservative) - Feature-only
   - [ ] Hybrid: Start with A, switch to B if needed

2. **Do you want me to start with Step A.1 (baseline)?**
   - This establishes your foundation (2-3 hours)
   - Gives you immediate feedback on LB

3. **Any preferences on models?**
   - Text: MiniLM (fast) vs MPNet (better)
   - Image: ResNet18 (fast) vs ResNet50 (better) vs EfficientNet (balanced)

4. **Should I create separate Python scripts for each step or one unified pipeline?**

Let me know and I'll start building! 🚀

---

## 🎯 SUCCESS CRITERIA

**Minimum Viable:** 40% SMAPE (better than 62%)  
**Target:** 30-40% SMAPE (your stated goal)  
**Stretch:** <30% SMAPE (top-tier competitive)

**Current Status:** 62.45% SMAPE → Need **1.5-2x improvement**

**Most Likely Path to Success:** Option A (Incremental) + Image Embeddings = 32-38% SMAPE

Let's do this! 💪
