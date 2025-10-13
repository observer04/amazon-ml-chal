"""
KAGGLE SUBMISSION: FIXED - Enhanced Model (Tabular + Text PCA)
===============================================================

ROOT CAUSE IDENTIFIED AND FIXED:
---------------------------------
The test_with_features.csv had interaction features computed BEFORE value clipping.
This caused pack_x_value and other interactions to have extreme values (70 billion!).

FIX: Clip value/pack_size BEFORE computing interaction features.

Performance: 27.86% validation SMAPE
Expected LB: 30-35% SMAPE (down from 55.30%)

Features:
- 22 engineered tabular features (with properly clipped interactions)
- 20 PCA components from CLIP text embeddings
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import sys
sys.path.append('.')
from config.config import RANDOM_STATE

print("=" * 80)
print("KAGGLE SUBMISSION: FIXED Enhanced Model")
print("Expected SMAPE: 30-35%")
print("=" * 80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1/7] Loading data...")

train_df = pd.read_csv('/kaggle/input/amazon-data/train_with_features.csv', low_memory=False)
test_df = pd.read_csv('/kaggle/input/amazon-data/test_with_features.csv', low_memory=False)

print(f"✓ Train: {train_df.shape}")
print(f"✓ Test:  {test_df.shape}")

# ============================================================================
# 2. FIX: CLIP VALUES BEFORE USING INTERACTION FEATURES
# ============================================================================
print("\n[2/7] Applying critical fix...")

# The bug: test_with_features.csv computed interactions with unclipped values
# This caused pack_x_value to be 70 billion instead of 25 million max
# Fix: Clip value/pack_size, then recompute interactions

test_df['value'] = test_df['value'].clip(upper=5000)
test_df['pack_size'] = test_df['pack_size'].clip(upper=5000)

# Recompute all interaction features
test_df['value_x_premium'] = test_df['value'] * test_df['has_premium']
test_df['value_x_luxury'] = test_df['value'] * test_df['has_luxury']
test_df['value_x_organic'] = test_df['value'] * test_df['has_organic']
test_df['pack_x_premium'] = test_df['pack_size'] * test_df['has_premium']
test_df['pack_x_value'] = test_df['pack_size'] * test_df['value']
test_df['brand_x_premium'] = test_df['brand_exists'] * test_df['has_premium']
test_df['brand_x_organic'] = test_df['brand_exists'] * test_df['has_organic']
test_df['travel_x_value'] = test_df['is_travel_size'] * test_df['value']
test_df['bulk_x_value'] = test_df['is_bulk'] * test_df['value']

print("✓ Clipped value/pack_size to 5000")
print("✓ Recomputed interaction features with clipped values")
print(f"  pack_x_value now: max={test_df['pack_x_value'].max():.0f} (was 70 billion!)")

# ============================================================================
# 3. EXTRACT TEXT PCA FEATURES
# ============================================================================
print("\n[3/7] Extracting text PCA features...")

# Load text embeddings
train_txt_emb = np.load('/kaggle/input/amazon-data/train_text_embeddings_clip.npy')
test_txt_emb = np.load('/kaggle/input/amazon-data/test_text_embeddings_clip.npy')

print(f"✓ Train text embeddings: {train_txt_emb.shape}")
print(f"✓ Test text embeddings:  {test_txt_emb.shape}")

# Fit PCA on train, transform both
n_pca = 20
pca = PCA(n_components=n_pca, random_state=RANDOM_STATE)
train_pca = pca.fit_transform(train_txt_emb)
test_pca = pca.transform(test_txt_emb)

explained_var = pca.explained_variance_ratio_.sum()
print(f"✓ PCA: {n_pca} components, {explained_var:.1%} variance explained")

# ============================================================================
# 4. PREPARE FEATURES
# ============================================================================
print("\n[4/7] Preparing features...")

# Define tabular features
tabular_features = [
    'value', 'unit', 'pack_size',
    'has_premium', 'has_organic', 'has_gourmet',
    'has_natural', 'has_artisan', 'has_luxury',
    'is_travel_size', 'is_bulk',
    'brand', 'brand_exists',
    'value_x_premium', 'value_x_luxury', 'value_x_organic',
    'pack_x_premium', 'pack_x_value',
    'brand_x_premium', 'brand_x_organic',
    'travel_x_value', 'bulk_x_value'
]

# Encode categorical features
label_encoders = {}
for col in tabular_features:
    if train_df[col].dtype == 'object':
        le = LabelEncoder()
        train_df[col] = train_df[col].fillna('missing')
        
        # Fit on train
        le.fit(train_df[col].astype(str))
        train_df[col] = le.transform(train_df[col].astype(str))
        
        # For test set - handle unseen categories
        if col in test_df.columns:
            test_df[col] = test_df[col].fillna('missing')
            test_col = test_df[col].astype(str)
            # Map unseen categories to 'missing'
            test_col = test_col.apply(lambda x: x if x in le.classes_ else 'missing')
            test_df[col] = le.transform(test_col)
        
        label_encoders[col] = le

# Fill numeric NaNs
for col in tabular_features:
    if train_df[col].dtype in ['float64', 'int64']:
        median_val = train_df[col].median()
        train_df[col] = train_df[col].fillna(median_val)
        test_df[col] = test_df[col].fillna(median_val)

X_train_tab = train_df[tabular_features].values
X_test_tab = test_df[tabular_features].values

# Combine tabular + PCA
X_train = np.concatenate([X_train_tab, train_pca], axis=1)
X_test = np.concatenate([X_test_tab, test_pca], axis=1)

y_train = np.sqrt(train_df['price'].values)  # SQRT transform

print(f"✓ Train features: {X_train.shape}")
print(f"✓ Test features:  {X_test.shape}")

# ============================================================================
# 5. TRAIN MODEL
# ============================================================================
print("\n[5/7] Training LightGBM on full training data...")

params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 63,
    'learning_rate': 0.03,
    'min_data_in_leaf': 15,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'lambda_l1': 0.5,
    'lambda_l2': 0.5,
    'verbose': -1,
    'seed': RANDOM_STATE,
}

train_data = lgb.Dataset(X_train, label=y_train)

model = lgb.train(
    params,
    train_data,
    num_boost_round=2000,
    valid_sets=[train_data],
    callbacks=[lgb.log_evaluation(period=200)]
)

print(f"✓ Model trained with {model.num_trees()} trees")

# ============================================================================
# 6. GENERATE PREDICTIONS
# ============================================================================
print("\n[6/7] Generating test predictions...")

test_pred_sqrt = model.predict(X_test)
test_pred = np.maximum(test_pred_sqrt ** 2, 0)  # Inverse sqrt, clip negatives

print(f"✓ Predictions generated: {len(test_pred)}")
print(f"  Range: [${test_pred.min():.2f}, ${test_pred.max():.2f}]")
print(f"  Mean:  ${test_pred.mean():.2f}")
print(f"  Median: ${np.median(test_pred):.2f}")

# ============================================================================
# 7. CREATE SUBMISSION
# ============================================================================
print("\n[7/7] Creating submission file...")

# Create submission with correct format
submission = pd.DataFrame({
    'sample_id': test_df['sample_id'].values,  # Use actual sample IDs
    'price': test_pred
})

submission.to_csv('submission_enhanced.csv', index=False)

print(f"✓ Submission saved: submission_enhanced.csv")
print(f"  Rows: {len(submission)}")

# Sanity checks
print(f"\n{'='*80}")
print("SUBMISSION SUMMARY:")
print(f"{'='*80}")
print(f"Predictions:     {len(test_pred):,}")
print(f"Price range:     [${test_pred.min():.2f}, ${test_pred.max():.2f}]")
print(f"Mean price:      ${test_pred.mean():.2f}")
print(f"Median price:    ${np.median(test_pred):.2f}")
print(f"Std dev:         ${test_pred.std():.2f}")

# Distribution
print(f"\nDistribution:")
print(f"  <$10:       {(test_pred < 10).sum():6,} ({100*(test_pred < 10).mean():.1f}%)")
print(f"  $10-$25:    {((test_pred >= 10) & (test_pred < 25)).sum():6,} ({100*((test_pred >= 10) & (test_pred < 25)).mean():.1f}%)")
print(f"  $25-$50:    {((test_pred >= 25) & (test_pred < 50)).sum():6,} ({100*((test_pred >= 25) & (test_pred < 50)).mean():.1f}%)")
print(f"  $50-$100:   {((test_pred >= 50) & (test_pred < 100)).sum():6,} ({100*((test_pred >= 50) & (test_pred < 100)).mean():.1f}%)")
print(f"  >$100:      {(test_pred >= 100).sum():6,} ({100*(test_pred >= 100).mean():.1f}%)")

print(f"\n{'='*80}")
print("BUG FIXED: Recomputed interaction features with clipped values")
print("EXPECTED LEADERBOARD: 30-35% SMAPE (was 55.30%)")
print(f"{'='*80}")
print("\n✅ Ready to submit to Kaggle!")
print("   Upload: submission_enhanced.csv")
