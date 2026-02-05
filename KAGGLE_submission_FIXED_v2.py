"""
KAGGLE SUBMISSION: FIXED v2 Enhanced Model (Frequency-encoded brands)
======================================================================

FIX: Use brand frequency encoding instead of label encoding
     48.5% of test samples have unseen brands -> encode as median frequency

Performance: 27.86% validation SMAPE
Expected LB: 30-35% SMAPE (after fix)

Features:
- 22 engineered tabular features (brand with frequency encoding)
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
print("KAGGLE SUBMISSION: FIXED v2 Enhanced Model")
print("Fix: Frequency encoding for brands")
print("=" * 80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1/6] Loading data...")

train_df = pd.read_csv('/kaggle/input/amazon-data/train_with_features.csv', low_memory=False)
test_df = pd.read_csv('/kaggle/input/amazon-data/test_with_features.csv', low_memory=False)

print(f"✓ Train: {train_df.shape}")
print(f"✓ Test:  {test_df.shape}")

# FIX: Clip extreme value outliers in test set
test_df['value'] = test_df['value'].clip(upper=5000)
test_df['pack_size'] = test_df['pack_size'].clip(upper=5000)

# ============================================================================
# 2. EXTRACT TEXT PCA FEATURES
# ============================================================================
print("\n[2/6] Extracting text PCA features...")

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
# 3. PREPARE FEATURES WITH FREQUENCY ENCODING
# ============================================================================
print("\n[3/6] Preparing features...")

# FIX: Frequency encode 'brand' instead of label encoding
print("\n  Applying frequency encoding to 'brand'...")

# Calculate brand frequencies from train
brand_freq = train_df['brand'].value_counts().to_dict()
median_freq = np.median(list(brand_freq.values()))

print(f"  Train brands: {len(brand_freq)}")
print(f"  Median frequency: {median_freq:.1f}")

# Encode train brands
train_df['brand_freq'] = train_df['brand'].map(brand_freq).fillna(median_freq)

# Encode test brands - unseen brands get median frequency
test_df['brand_freq'] = test_df['brand'].map(brand_freq).fillna(median_freq)

# Check how many test brands are unseen
test_unseen = test_df['brand'].map(brand_freq).isna().sum()
print(f"  Test samples with unseen brands: {test_unseen} ({100*test_unseen/len(test_df):.1f}%)")
print(f"  → Encoded with median frequency: {median_freq:.1f}")

# Define tabular features - use brand_freq instead of brand
tabular_features = [
    'value', 'unit', 'pack_size',
    'has_premium', 'has_organic', 'has_gourmet',
    'has_natural', 'has_artisan', 'has_luxury',
    'is_travel_size', 'is_bulk',
    'brand_freq', 'brand_exists',  # brand_freq instead of brand
    'value_x_premium', 'value_x_luxury', 'value_x_organic',
    'pack_x_premium', 'pack_x_value',
    'brand_x_premium', 'brand_x_organic',
    'travel_x_value', 'bulk_x_value'
]

print(f"✓ Using {len(tabular_features)} tabular features (brand_freq encoding)")

# Encode categorical features (only 'unit' now)
label_encoders = {}
for col in tabular_features:
    if col == 'brand_freq':
        continue  # Skip - already encoded
        
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
        print(f"  Encoded '{col}': {len(le.classes_)} unique values")

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
# 4. TRAIN MODEL
# ============================================================================
print("\n[4/6] Training LightGBM on full training data...")

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
# 5. GENERATE PREDICTIONS
# ============================================================================
print("\n[5/6] Generating test predictions...")

test_pred_sqrt = model.predict(X_test)
test_pred = np.maximum(test_pred_sqrt ** 2, 0)  # Inverse sqrt, clip negatives

print(f"✓ Predictions generated: {len(test_pred)}")
print(f"  Range: [${test_pred.min():.2f}, ${test_pred.max():.2f}]")
print(f"  Mean:  ${test_pred.mean():.2f}")
print(f"  Median: ${np.median(test_pred):.2f}")

# ============================================================================
# 6. CREATE SUBMISSION
# ============================================================================
print("\n[6/6] Creating submission file...")

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
print("FIX APPLIED: Frequency encoding for brands (handles unseen brands)")
print("EXPECTED LEADERBOARD: 30-35% SMAPE")
print(f"{'='*80}")
print("\n✅ Ready to submit to Kaggle!")
print("   Upload: submission_enhanced.csv")
