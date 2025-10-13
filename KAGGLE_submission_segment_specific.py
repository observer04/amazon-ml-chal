"""
ENHANCEMENT 2: Segment-Specific Models
=======================================
Train separate models for Budget/Mid/Premium/Luxury segments
Expected: 0.8-1.2% improvement (27.86% → 26.7%)
Runtime: +15 min

REASONING:
- Budget (<$10):     37.44% SMAPE - needs help!
- Mid-Range ($10-50): 19.01% SMAPE - already good
- Premium ($50-100):  32.32% SMAPE - needs tuning
- Luxury (>$100):     46.14% SMAPE - small segment
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

print("="*80)
print("ENHANCEMENT 2: Segment-Specific Models")
print("="*80)

# Load data
train_df = pd.read_csv('/kaggle/input/amazon-data/train_with_features.csv', low_memory=False)
test_df = pd.read_csv('/kaggle/input/amazon-data/test_with_features.csv', low_memory=False)
test_df['value'] = test_df['value'].clip(upper=5000)

# Load embeddings
train_txt_emb = np.load('/kaggle/input/amazon-data/train_text_embeddings_clip.npy')
test_txt_emb = np.load('/kaggle/input/amazon-data/test_text_embeddings_clip.npy')

# PCA
pca = PCA(n_components=20, random_state=42)
train_pca = pca.fit_transform(train_txt_emb)
test_pca = pca.transform(test_txt_emb)

# Features
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

# Encode
label_encoders = {}
for col in tabular_features:
    if train_df[col].dtype == 'object':
        le = LabelEncoder()
        train_df[col] = train_df[col].fillna('missing')
        test_df[col] = test_df[col].fillna('missing')
        
        combined = pd.concat([train_df[col], test_df[col]]).astype(str)
        le.fit(combined)
        
        train_df[col] = le.transform(train_df[col].astype(str))
        test_df[col] = le.transform(test_df[col].astype(str))

# Combine features
X_train = np.concatenate([train_df[tabular_features].values, train_pca], axis=1)
X_test = np.concatenate([test_df[tabular_features].values, test_pca], axis=1)

# Define segments based on VALUE (proxy for price)
def get_segment_mask(df):
    """Segment by value as proxy for price"""
    value = df['value'].fillna(0)
    budget = value < 10
    mid = (value >= 10) & (value < 50)
    premium = (value >= 50) & (value < 100)
    luxury = value >= 100
    return budget, mid, premium, luxury

train_segments = get_segment_mask(train_df)
test_segments = get_segment_mask(test_df)

# Train segment-specific models
segment_names = ['Budget', 'Mid-Range', 'Premium', 'Luxury']
segment_params = {
    'Budget': {'num_leaves': 31, 'learning_rate': 0.05},  # Simpler model
    'Mid-Range': {'num_leaves': 63, 'learning_rate': 0.03},  # Default
    'Premium': {'num_leaves': 95, 'learning_rate': 0.02},  # More complex
    'Luxury': {'num_leaves': 15, 'learning_rate': 0.1},  # Small data, simple model
}

test_predictions = np.zeros(len(test_df))

for i, (seg_name, train_mask, test_mask) in enumerate(zip(segment_names, train_segments, test_segments)):
    print(f"\n[{i+1}/4] Training {seg_name} model...")
    print(f"  Train samples: {train_mask.sum():,}")
    print(f"  Test samples:  {test_mask.sum():,}")
    
    if train_mask.sum() < 100:
        print(f"  ⚠️  Too few samples, using global model")
        continue
    
    # Segment data
    X_train_seg = X_train[train_mask]
    y_train_seg = np.sqrt(train_df.loc[train_mask, 'price'].values)
    X_test_seg = X_test[test_mask]
    
    # Segment-specific params
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': segment_params[seg_name]['num_leaves'],
        'learning_rate': segment_params[seg_name]['learning_rate'],
        'min_data_in_leaf': 15,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'lambda_l1': 0.5,
        'lambda_l2': 0.5,
        'verbose': -1,
        'seed': 42,
    }
    
    # Train
    train_data = lgb.Dataset(X_train_seg, label=y_train_seg)
    model = lgb.train(params, train_data, num_boost_round=1500)
    
    # Predict
    pred_sqrt = model.predict(X_test_seg)
    pred = np.maximum(pred_sqrt ** 2, 0)
    test_predictions[test_mask] = pred
    
    print(f"  ✓ {seg_name}: mean=${pred.mean():.2f}, median=${np.median(pred):.2f}")

# Handle any missing predictions (use global model fallback)
missing_mask = test_predictions == 0
if missing_mask.sum() > 0:
    print(f"\n⚠️  {missing_mask.sum()} predictions missing, using global model fallback")
    # Use your original submission_enhanced.csv values
    fallback = pd.read_csv('submission_enhanced.csv')
    test_predictions[missing_mask] = fallback.loc[missing_mask, 'price'].values

# Create submission
print(f"\nFinal predictions:")
print(f"  Mean:   ${test_predictions.mean():.2f}")
print(f"  Median: ${np.median(test_predictions):.2f}")
print(f"  Range:  [${test_predictions.min():.2f}, ${test_predictions.max():.2f}]")

submission = pd.DataFrame({
    'sample_id': test_df['sample_id'],
    'price': test_predictions
})
submission.to_csv('submission_segment_specific.csv', index=False)
print(f"\n✅ Saved: submission_segment_specific.csv")
