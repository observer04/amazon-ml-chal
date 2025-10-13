# IMAGE FEATURE HYPOTHESIS PLAN
**Date:** October 13, 2025  
**Goal:** Improve from 57.75% → 45% SMAPE using product images

## DISCOVERED MODELS

### 1. **Marqo E-commerce Embeddings** ⭐⭐⭐⭐⭐
- **Model:** `Marqo/marqo-ecommerce-embeddings-L` (652M params, 1024-dim)
- **Benchmark:** 38.9% better than Amazon-Titan-Multimodal on Amazon products
- **Trained on:** Amazon Products (3M) + Google Shopping (1M)
- **Perfect fit:** Specifically trained for e-commerce product price prediction
- **Size:** 652M params (may fit in Kaggle)

### 2. **Trendyol E-commerce Encoder**
- **Model:** `Trendyol/e-commerce-product-image-encoder`
- **Base:** ConvNext-Base fine-tuned on e-commerce
- **Use case:** Product image classification

### 3. **Image Quality/Aesthetic Models**
- **Aesthetic Scorer:** `rsinema/aesthetic-scorer` (7 metrics: quality, composition, lighting, etc.)
- **Quality Fusion:** `matthewyuan/image-quality-fusion` (BRISQUE + aesthetic + CLIP)
- **LAION Aesthetic:** Aesthetic scoring for images

## HYPOTHESIS STRATEGIES

### **Strategy 1: Marqo E-commerce Embeddings** 🏆 TOP PRIORITY
**Hypothesis:** CLIP failed because it's generic. Marqo is trained specifically on Amazon products for e-commerce tasks.

**Implementation:**
```python
from transformers import AutoModel, AutoProcessor
import torch
from PIL import Image
import requests

model_name = 'Marqo/marqo-ecommerce-embeddings-L'
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

# Download images from URLs and process
for url in image_urls:
    img = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    processed = processor(images=[img], return_tensors="pt")
    
    with torch.no_grad():
        image_features = model.get_image_features(**processed)
        # 1024-dim embeddings
```

**Expected improvement:** 5-10% SMAPE (could be game-changer)
**Risk:** Model size (652M params) might exceed Kaggle limits
**Test locally first:** Before deploying to Kaggle

---

### **Strategy 2: Image Quality Features** ⭐⭐⭐⭐
**Hypothesis:** Product photo quality correlates with price. Professional photos → higher price.

**Implementation:**
```python
from aesthetic_scorer import AestheticScorer
from transformers import CLIPProcessor

# Load aesthetic scorer
processor = CLIPProcessor.from_pretrained("rsinema/aesthetic-scorer")
model = torch.load("rsinema/aesthetic-scorer/model.pt")

# Extract 7 aesthetic features per image:
# - Overall aesthetic score
# - Technical quality
# - Composition
# - Lighting
# - Color harmony
# - Depth of field
# - Content score

aesthetic_features = []
for img in images:
    inputs = processor(images=img, return_tensors="pt")["pixel_values"]
    with torch.no_grad():
        scores = model(inputs)  # 7 scores (0-5 each)
    aesthetic_features.append(scores.numpy())
```

**Expected improvement:** 2-4% SMAPE
**Pros:** Lightweight, interpretable features
**Cons:** May need image downloads

---

### **Strategy 3: Multi-Modal Fusion** ⭐⭐⭐⭐⭐
**Hypothesis:** Combine Marqo image + text embeddings with better fusion than simple concatenation.

**Implementation:**
```python
# Get both image and text features from Marqo
image_features = model.get_image_features(**image_inputs)  # 1024-dim
text_features = model.get_text_features(**text_inputs)      # 1024-dim

# Cross-attention fusion
attention_weights = (image_features @ text_features.T).softmax(dim=-1)
fused_features = attention_weights @ text_features

# Or: Gated fusion
gate = torch.sigmoid(W_gate @ torch.cat([image_features, text_features]))
fused = gate * image_features + (1 - gate) * text_features
```

**Expected improvement:** 8-12% SMAPE
**Pros:** Leverages both modalities optimally
**Cons:** More complex, needs tuning

---

### **Strategy 4: Image Metadata Features** ⭐⭐⭐
**Hypothesis:** Image properties (resolution, aspect ratio, file size) indicate listing quality.

**Implementation:**
```python
from PIL import Image
import requests
from io import BytesIO

metadata_features = []
for url in image_urls:
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    
    features = {
        'width': img.width,
        'height': img.height,
        'aspect_ratio': img.width / img.height,
        'resolution': img.width * img.height,
        'file_size': len(response.content),
        'format': img.format,
        'mode': img.mode,
        'has_transparency': img.mode in ['RGBA', 'LA', 'P']
    }
    metadata_features.append(features)
```

**Expected improvement:** 1-2% SMAPE
**Pros:** Very fast, no model needed
**Cons:** Weak signal, but FREE to compute

---

### **Strategy 5: Ensemble of All Above** ⭐⭐⭐⭐⭐
**Hypothesis:** Combining all image signals gives best performance.

**Features (total ~1040-dim):**
- Marqo image embeddings: 1024-dim
- Aesthetic scores: 7-dim
- Image metadata: 8-dim
- Combined with existing text (20-dim) + tabular (22-dim)

**Expected improvement:** 10-15% SMAPE (could reach 45% target!)

---

## IMPLEMENTATION PLAN

### Phase 1: Quick Wins (2 hours)
1. **Extract image metadata** (30 min)
   - Download sample of 1000 images
   - Extract resolution, size, aspect ratio
   - Test R² correlation with price
   
2. **Test aesthetic scorer** (1.5 hours)
   - Download aesthetic model
   - Process 1000 sample images
   - Check correlation with price
   - If R² > 0.01, process all images

### Phase 2: Marqo Integration (3-4 hours)
1. **Download Marqo model** (30 min)
   - Check model size fits in Kaggle
   - Test on 100 images first
   
2. **Process all images** (2 hours)
   - 75K train + 75K test images
   - Save embeddings to `.npy`
   
3. **Test embeddings** (1 hour)
   - Check R² vs CLIP (should be much better)
   - Test in LightGBM model
   - Quick CV validation

### Phase 3: Model Training (2-3 hours)
1. **Feature engineering**
   - Combine Marqo + aesthetic + metadata
   - Keep existing text + tabular
   
2. **Model training**
   - 5-fold CV with full features
   - Target: < 50% SMAPE
   
3. **Kaggle submission**
   - Create submission script
   - Submit and monitor LB

---

## RESOURCE REQUIREMENTS

### Kaggle Notebook Limits:
- **GPU:** T4 (16GB VRAM) ✅ Should fit Marqo-L (652M)
- **RAM:** 13GB ✅ Enough for embeddings
- **Time:** 9 hours ✅ Enough for inference
- **Disk:** 20GB ✅ Enough for model + data

### Image Downloads:
- **Count:** 150K images (75K train + 75K test)
- **Size:** ~50MB total (estimated 350KB each)
- **Time:** ~30-45 minutes with parallel requests

---

## EXPECTED OUTCOMES

### Conservative Estimate:
- Marqo embeddings: +5% improvement → 52.75%
- Aesthetic features: +2% improvement → 50.75%
- Metadata: +1% improvement → 49.75%
- **Final:** ~50% SMAPE (rank ~1500)

### Optimistic Estimate:
- Marqo + multi-modal fusion: +10% → 47.75%
- Aesthetic + metadata: +2% → 45.75%
- **Final:** ~45% SMAPE (rank ~500-800)

### Best Case:
- Perfect implementation + ensemble → **40-42% SMAPE** (top 100)

---

## NEXT STEPS

**IMMEDIATE:**
1. Test Marqo model size (can it fit in Kaggle?)
2. Download 1000 sample images
3. Extract Marqo embeddings for 1000 images
4. Check correlation with price
5. If R² > 0.05 → PROCEED WITH FULL EXTRACTION

**Print next command needed.**
