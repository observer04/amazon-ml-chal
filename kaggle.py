"""
=================================================================================
KAGGLE SUBMISSION CODE - ML CHALLENGE 2025
Amazon Product Pricing Prediction with Hybrid Ensemble
=================================================================================

This is the main training and prediction script for Kaggle P100 GPU.
Import stable modules from .py files and run complete pipeline.

Architecture: Hybrid Ensemble
  - Model 1: LightGBM (55%) - Tabular features (IPQ, quality keywords, text stats)
  - Model 2: Text MLP (25%) - 384-dim sentence-transformers embeddings
  - Model 3: Image MLP (20%) - 512-dim ResNet18 embeddings

Evaluation Metric: SMAPE (Symmetric Mean Absolute Percentage Error)
Target: 8-12% SMAPE for competitive leaderboard position
=================================================================================
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Import stable modules
from src.feature_extraction import create_features, get_tabular_features
from src.text_embeddings import TextEmbedder, create_text_embeddings
from src.image_processing import ImageEmbedder, create_image_embeddings
from src.models import cv_train_lightgbm, cv_train_mlp, predict_lightgbm, predict_mlp
from src.ensemble import Ensemble, smape, evaluate_model, cv_ensemble_predictions
from config.config import *

print("="*80)
print("ML CHALLENGE 2025 - AMAZON PRICING PREDICTION")
print("="*80)

# =============================================================================
# STEP 1: LOAD DATA
# =============================================================================
print("\n[1/8] Loading data...")

train_df = pd.read_csv(f'{DATA_PATH}/train.csv')
test_df = pd.read_csv(f'{DATA_PATH}/test.csv')

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"Target (price) range: ${train_df['price'].min():.2f} - ${train_df['price'].max():.2f}")
print(f"Target (price) mean: ${train_df['price'].mean():.2f}, median: ${train_df['price'].median():.2f}")

# =============================================================================
# STEP 2: FEATURE ENGINEERING
# =============================================================================
print("\n[2/8] Engineering features...")

# Extract all structured features from catalog_content
train_features = create_features(train_df.copy())
test_features = create_features(test_df.copy())

print(f"Train features shape: {train_features.shape}")
print(f"Test features shape: {test_features.shape}")

# Get feature names for LightGBM (numeric only)
tabular_feature_cols = get_tabular_features(train_features)
print(f"Tabular features for LightGBM: {len(tabular_feature_cols)}")
print(f"Features: {tabular_feature_cols}")

# Prepare tabular data
X_tabular = train_features[tabular_feature_cols].fillna(0)
y = train_features['price']

X_tabular_test = test_features[tabular_feature_cols].fillna(0)

# =============================================================================
# STEP 3: TEXT EMBEDDINGS
# =============================================================================
print("\n[3/8] Generating text embeddings...")

text_embedder = TextEmbedder(device='cuda')  # Use GPU

# Generate embeddings from catalog_content
train_text_embeddings = create_text_embeddings(
    train_df, 
    text_column='catalog_content',
    embedder=text_embedder,
    save_path='outputs/train_text_embeddings.npy'
)

test_text_embeddings = create_text_embeddings(
    test_df,
    text_column='catalog_content', 
    embedder=text_embedder,
    save_path='outputs/test_text_embeddings.npy'
)

print(f"Train text embeddings: {train_text_embeddings.shape}")
print(f"Test text embeddings: {test_text_embeddings.shape}")

# =============================================================================
# STEP 4: IMAGE EMBEDDINGS
# =============================================================================
print("\n[4/8] Generating image embeddings...")

image_embedder = ImageEmbedder(model_name='resnet18', device='cuda')  # Use GPU

# Generate embeddings from image URLs
train_image_embeddings = create_image_embeddings(
    train_df,
    url_column='image_link',
    embedder=image_embedder,
    save_path='outputs/train_image_embeddings.npy'
)

test_image_embeddings = create_image_embeddings(
    test_df,
    url_column='image_link',
    embedder=image_embedder,
    save_path='outputs/test_image_embeddings.npy'
)

print(f"Train image embeddings: {train_image_embeddings.shape}")
print(f"Test image embeddings: {test_image_embeddings.shape}")

# =============================================================================
# STEP 5: TRAIN MODELS WITH CROSS-VALIDATION
# =============================================================================
print("\n[5/8] Training models with 5-fold CV...")

# 5.1: LightGBM on tabular features
print("\n--- Training LightGBM ---")
lgb_models, lgb_oof = cv_train_lightgbm(X_tabular, y, n_splits=N_SPLITS)
evaluate_model(y, lgb_oof, "LightGBM OOF")

# 5.2: Text MLP on text embeddings
print("\n--- Training Text MLP ---")
text_mlp_models, text_scalers, text_oof = cv_train_mlp(
    train_text_embeddings, 
    y.values, 
    n_splits=N_SPLITS
)
evaluate_model(y, text_oof, "Text MLP OOF")

# 5.3: Image MLP on image embeddings
print("\n--- Training Image MLP ---")
image_mlp_models, image_scalers, image_oof = cv_train_mlp(
    train_image_embeddings,
    y.values,
    n_splits=N_SPLITS
)
evaluate_model(y, image_oof, "Image MLP OOF")

# =============================================================================
# STEP 6: OPTIMIZE ENSEMBLE WEIGHTS
# =============================================================================
print("\n[6/8] Optimizing ensemble weights...")

# Create OOF predictions dict
oof_predictions = {
    'lightgbm': lgb_oof,
    'text_mlp': text_oof,
    'image_mlp': image_oof
}

# Initialize ensemble and optimize weights
ensemble = Ensemble(weights=ENSEMBLE_WEIGHTS)
optimized_weights = ensemble.optimize_weights(oof_predictions, y.values)

# Evaluate ensemble on OOF
ensemble_oof = ensemble.predict(oof_predictions)
evaluate_model(y, ensemble_oof, "Ensemble OOF")

# =============================================================================
# STEP 7: GENERATE TEST PREDICTIONS
# =============================================================================
print("\n[7/8] Generating test predictions...")

# Predict with each model (average across CV folds)
test_lgb_preds = np.mean([predict_lightgbm(model, X_tabular_test) for model in lgb_models], axis=0)
print(f"LightGBM test preds: mean={test_lgb_preds.mean():.2f}, std={test_lgb_preds.std():.2f}")

test_text_preds = np.mean([
    predict_mlp(model, test_text_embeddings, scaler)
    for model, scaler in zip(text_mlp_models, text_scalers)
], axis=0)
print(f"Text MLP test preds: mean={test_text_preds.mean():.2f}, std={test_text_preds.std():.2f}")

test_image_preds = np.mean([
    predict_mlp(model, test_image_embeddings, scaler)
    for model, scaler in zip(image_mlp_models, image_scalers)
], axis=0)
print(f"Image MLP test preds: mean={test_image_preds.mean():.2f}, std={test_image_preds.std():.2f}")

# Ensemble predictions
test_predictions_dict = {
    'lightgbm': test_lgb_preds,
    'text_mlp': test_text_preds,
    'image_mlp': test_image_preds
}

final_test_preds = ensemble.predict(test_predictions_dict)
print(f"Ensemble test preds: mean={final_test_preds.mean():.2f}, std={final_test_preds.std():.2f}")

# Clip predictions to reasonable range (avoid negative prices)
final_test_preds = np.clip(final_test_preds, 0.01, 3000)

# =============================================================================
# STEP 8: CREATE SUBMISSION
# =============================================================================
print("\n[8/8] Creating submission file...")

submission = pd.DataFrame({
    'sample_id': test_df['sample_id'],
    'price': final_test_preds
})

submission.to_csv('submission.csv', index=False)
print(f"Submission saved: submission.csv")
print(f"Submission shape: {submission.shape}")
print(f"\nSample predictions:")
print(submission.head(10))

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)
print(f"OOF SMAPE: {smape(y.values, ensemble_oof):.4f}")
print(f"Ensemble weights: {ensemble.weights}")
print(f"Submission file: submission.csv")
print("="*80)
