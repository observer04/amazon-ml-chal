# ML Challenge 2025 - Amazon Product Pricing Prediction

**Goal**: Predict product prices from catalog text and images using hybrid ensemble approach.

**Evaluation Metric**: SMAPE (Symmetric Mean Absolute Percentage Error)

**Target Performance**: 8-12% SMAPE for competitive leaderboard position

---

## 🏗️ Architecture

**Hybrid Ensemble Model** combining three specialized predictors:

1. **LightGBM (55% weight)** - Tabular features
   - Value & Unit extraction (IPQ)
   - Quality keywords (Premium, Organic, Gourmet, Natural, Artisan, Luxury)
   - Brand extraction
   - Text statistics (length, word count, bullet structure)
   - Pack size detection

2. **Text MLP (25% weight)** - Semantic embeddings
   - 384-dim sentence-transformers embeddings (all-MiniLM-L6-v2)
   - 2-layer MLP: [128, 64] → 1 (regression)
   - Dropout 0.3, early stopping

3. **Image MLP (20% weight)** - Visual embeddings
   - 512-dim ResNet18 embeddings (pre-trained on ImageNet)
   - 2-layer MLP: [128, 64] → 1 (regression)
   - Dropout 0.3, early stopping

**Ensemble**: Weighted average with optimized weights (scipy SLSQP optimization)

---

## 📊 Key Insights from EDA

- **IPQ is PRIMARY predictor**: 98.7% of products have Value+Unit (e.g., "12 Ounce")
- **Quality keywords show massive premiums**:
  - Premium: +46.6% price premium
  - Gourmet: +39.2% premium
  - Organic: +28.4% premium
- **Low-price items (<$10) are high-risk** due to SMAPE metric penalizing relative errors
- **Bullet points alone are weak**, but structure matters when combined with other features

---

## 📁 Repository Structure

```
amazonchal/
├── kaggle.py                    # Main training script (copy to Kaggle)
├── requirements.txt             # Dependencies
├── README.md                    # This file
├── process.md                   # Methodology documentation
├── comprehensive_eda.ipynb      # Exploratory data analysis
│
├── config/
│   └── config.py                # Hyperparameters and settings
│
├── src/
│   ├── feature_extraction.py   # Parse catalog_content, extract IPQ, brand, keywords
│   ├── text_embeddings.py      # Generate sentence-transformers embeddings
│   ├── image_processing.py     # ResNet18 feature extraction from URLs
│   ├── models.py                # LightGBM & MLP implementations with CV
│   └── ensemble.py              # Weighted ensemble & SMAPE optimization
│
├── dataset/
│   ├── train.csv                # Training data (75k products)
│   ├── test.csv                 # Test data for submission
│   └── train_with_features.csv  # Pre-computed features from EDA
│
└── outputs/
    ├── train_text_embeddings.npy    # Cached embeddings (384-dim)
    ├── test_text_embeddings.npy
    ├── train_image_embeddings.npy   # Cached embeddings (512-dim)
    └── test_image_embeddings.npy
```

---

## 🚀 Quick Start (Kaggle Deployment)

### Option 1: Run on Kaggle GPU

1. **Upload files to Kaggle**:
   - Upload all files from `src/` and `config/` folders
   - Create new notebook

2. **Install dependencies** (in Kaggle notebook):
   ```python
   !pip install sentence-transformers lightgbm
   ```

3. **Copy-paste `kaggle.py`** into Kaggle notebook cell

4. **Run** (should take ~2-3 hours on P100 GPU):
   ```python
   # Adjust DATA_PATH in config/config.py to match Kaggle input path
   # Then run the entire kaggle.py script
   exec(open('kaggle.py').read())
   ```

5. **Submit `submission.csv`** to leaderboard

### Option 2: Run Locally

1. **Create virtual environment**:
   ```bash
   python3.12 -m venv amzml
   source amzml/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Update paths** in `config/config.py`:
   ```python
   DATA_PATH = 'dataset'  # Local path
   OUTPUT_PATH = 'outputs'
   ```

4. **Run training**:
   ```bash
   python kaggle.py
   ```

---

## 🔧 Configuration

Edit `config/config.py` to tune hyperparameters:

```python
# Model weights (will be optimized during training)
ENSEMBLE_WEIGHTS = {
    'lightgbm': 0.55,
    'text_mlp': 0.25,
    'image_mlp': 0.20,
}

# LightGBM
LIGHTGBM_PARAMS = {
    'objective': 'mape',
    'num_leaves': 31,
    'learning_rate': 0.05,
    ...
}

# Neural networks
MLP_HIDDEN_DIMS = [128, 64]
LEARNING_RATE = 0.001
BATCH_SIZE = 256
EPOCHS = 50
```

---

## 📈 Expected Performance

Based on EDA and model architecture:

- **LightGBM alone**: ~12-15% SMAPE (strong baseline with IPQ features)
- **Text MLP alone**: ~18-22% SMAPE (semantic understanding)
- **Image MLP alone**: ~22-28% SMAPE (visual features, many failed downloads)
- **Hybrid Ensemble**: **8-12% SMAPE** (target, leverages all modalities)

---

## 🧪 Validation Strategy

- **5-Fold Cross-Validation** with stratified splits
- **Out-of-Fold (OOF) predictions** for ensemble weight optimization
- **SMAPE minimization** using scipy.optimize (SLSQP algorithm)
- **Test predictions**: Average of 5 fold models

---

## 🛠️ Troubleshooting

### Issue: Import errors in Kaggle
**Solution**: Make sure `src/` and `config/` folders are in the same directory as `kaggle.py`. Alternatively, copy-paste module code directly into notebook.

### Issue: Image download failures
**Solution**: This is expected. The model uses zero vectors for failed images. Image MLP still provides ~20% boost to ensemble.

### Issue: Out of memory (OOM)
**Solution**: Reduce `BATCH_SIZE` in `config/config.py` (try 128 or 64).

### Issue: Training takes too long
**Solution**: 
- Reduce `EPOCHS` to 30
- Use fewer CV folds: `N_SPLITS = 3`
- Skip image processing (set weight to 0)

---

## 📝 Citation

If you found this approach useful:

```
ML Challenge 2025 - Amazon Product Pricing
Hybrid Ensemble: LightGBM + Text Embeddings + Image CNN
Author: ML Challenger
```

---

## 📞 Contact

For questions or issues, please open a GitHub issue or contact via Kaggle discussion.

---

**Good luck on the leaderboard! 🚀**

---
---

# ORIGINAL PROBLEM STATEMENT

## Smart Product Pricing Challenge

In e-commerce, determining the optimal price point for products is crucial for marketplace success and customer satisfaction. Your challenge is to develop an ML solution that analyzes product details and predict the price of the product. The relationship between product attributes and pricing is complex - with factors like brand, specifications, product quantity directly influence pricing. Your task is to build a model that can analyze these product details holistically and suggest an optimal price.

### Data Description:

The dataset consists of the following columns:

1. **sample_id:** A unique identifier for the input sample
2. **catalog_content:** Text field containing title, product description and an Item Pack Quantity(IPQ) concatenated.
3. **image_link:** Public URL where the product image is available for download. 
   Example link - https://m.media-amazon.com/images/I/71XfHPR36-L.jpg
   To download images use `download_images` function from `src/utils.py`. See sample code in `src/test.ipynb`.
4. **price:** Price of the product (Target variable - only available in training data)

### Dataset Details:

- **Training Dataset:** 75k products with complete product details and prices
- **Test Set:** 75k products for final evaluation

### Output Format:

The output file should be a CSV with 2 columns:

1. **sample_id:** The unique identifier of the data sample. Note the ID should match the test record sample_id.
2. **price:** A float value representing the predicted price of the product.

Note: Make sure to output a prediction for all sample IDs. If you have less/more number of output samples in the output file as compared to test.csv, your output won't be evaluated.

### File Descriptions:

*Source files*

1. **src/utils.py:** Contains helper functions for downloading images from the image_link. You may need to retry a few times to download all images due to possible throttling issues.
2. **sample_code.py:** Sample dummy code that can generate an output file in the given format. Usage of this file is optional.

*Dataset files*

1. **dataset/train.csv:** Training file with labels (`price`).
2. **dataset/test.csv:** Test file without output labels (`price`). Generate predictions using your model/solution on this file's data and format the output file to match sample_test_out.csv
3. **dataset/sample_test.csv:** Sample test input file.
4. **dataset/sample_test_out.csv:** Sample outputs for sample_test.csv. The output for test.csv must be formatted in the exact same way. Note: The predictions in the file might not be correct

### Constraints:

1. You will be provided with a sample output file. Format your output to match the sample output file exactly. 

2. Predicted prices must be positive float values.

3. Final model should be a MIT/Apache 2.0 License model and up to 8 Billion parameters.

### Evaluation Criteria:

Submissions are evaluated using **Symmetric Mean Absolute Percentage Error (SMAPE)**: A statistical measure that expresses the relative difference between predicted and actual values as a percentage, while treating positive and negative errors equally.

**Formula:**
```
SMAPE = (1/n) * Σ |predicted_price - actual_price| / ((|actual_price| + |predicted_price|)/2)
```

**Example:** If actual price = $100 and predicted price = $120  
SMAPE = |100-120| / ((|100| + |120|)/2) * 100% = 18.18%

**Note:** SMAPE is bounded between 0% and 200%. Lower values indicate better performance.

### Leaderboard Information:

- **Public Leaderboard:** During the challenge, rankings will be based on 25K samples from the test set to provide real-time feedback on your model's performance.
- **Final Rankings:** The final decision will be based on performance on the complete 75K test set along with provided documentation of the proposed approach by the teams.

### Submission Requirements:

1. Upload a `test_out.csv` file in the Portal with the exact same formatting as `sample_test_out.csv`

2. All participating teams must also provide a 1-page document describing:
   - Methodology used
   - Model architecture/algorithms selected
   - Feature engineering techniques applied
   - Any other relevant information about the approach
   Note: A sample template for this documentation is provided in Documentation_template.md

### **Academic Integrity and Fair Play:**

**⚠️ STRICTLY PROHIBITED: External Price Lookup**

Participants are **STRICTLY NOT ALLOWED** to obtain prices from the internet, external databases, or any sources outside the provided dataset. This includes but is not limited to:
- Web scraping product prices from e-commerce websites
- Using APIs to fetch current market prices
- Manual price lookup from online sources
- Using any external pricing databases or services

**Enforcement:**
- All submitted approaches, methodologies, and code pipelines will be thoroughly reviewed and verified
- Any evidence of external price lookup or data augmentation from internet sources will result in **immediate disqualification**

**Fair Play:** This challenge is designed to test your machine learning and data science skills using only the provided training data. External price lookup defeats the purpose of the challenge.


### Tips for Success:

- Consider both textual features (catalog_content) and visual features (product images)
- Explore feature engineering techniques for text and image data
- Consider ensemble methods combining different model types
- Pay attention to outliers and data preprocessing
