"""
KAGGLE SUBMISSION: Enhanced Model (Tabular + Text PCA)
=======================================================

Performance: 27.86% validation SMAPE
Expected LB: 27-30% SMAPE

Features:
- 22 engineered tabular features
- 20 PCA components from CLIP text embeddings
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import pickle
import sys
sys.path.append('.')
from config.config import RANDOM_STATE

print("=" * 80)
print("KAGGLE SUBMISSION: Enhanced Model")
print("Expected SMAPE: 27-30%")
print("=" * 80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1/6] Loading data...")

train_df = pd.read_csv('/kaggle/working/amazon-ml-chal/dataset/train_with_features.csv', low_memory=False)
test_df = pd.read_csv('/kaggle/working/amazon-ml-chal/dataset/test.csv', low_memory=False)

print(f"✓ Train: {train_df.shape}")
print(f"✓ Test:  {test_df.shape}")

# ============================================================================
# 2. EXTRACT TEXT PCA FEATURES
# ============================================================================
print("\n[2/6] Extracting text PCA features...")

# Load text embeddings
train_txt_emb = np.load('/kaggle/working/amazon-ml-chal/outputs/train_text_embeddings_clip.npy')
test_txt_emb = np.load('/kaggle/working/amazon-ml-chal/outputs/test_text_embeddings_clip.npy')

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
# 3. PREPARE FEATURES
# ============================================================================
print("\n[3/6] Preparing features...")

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
        
        # Fit on train, but handle unseen categories in test
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

# Extract tabular features
# For test set, we need to generate the features first
# Check if test_df has these features already
if 'value' not in test_df.columns:
    print("⚠️  Test set doesn't have engineered features!")
    print("   Need to run feature extraction on test set first.")
    print("   Loading from test_with_features.csv if available...")
    
    try:
        test_df = pd.read_csv('/kaggle/working/amazon-ml-chal/dataset/test_with_features.csv', low_memory=False)
        print(f"✓ Loaded test_with_features.csv: {test_df.shape}")
        
        # Encode categorical for test
        for col in tabular_features:
            if col in label_encoders:
                test_df[col] = test_df[col].fillna('missing')
                test_col = test_df[col].astype(str)
                test_col = test_col.apply(lambda x: x if x in label_encoders[col].classes_ else 'missing')
                test_df[col] = label_encoders[col].transform(test_col)
    except:
        print("❌ test_with_features.csv not found!")
        print("   Please run generate_features.py on test set first")
        sys.exit(1)

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

submission = pd.DataFrame({
    'id': range(len(test_pred)),
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
print("EXPECTED LEADERBOARD: 27-30% SMAPE")
print(f"{'='*80}")
print("\n✅ Ready to submit to Kaggle!")
print("   Upload: submission_enhanced.csv")
