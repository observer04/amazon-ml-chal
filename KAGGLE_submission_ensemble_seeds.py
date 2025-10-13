"""
ENHANCEMENT 1: Ensemble Multiple Seeds
=======================================
Train 3-5 models with different random seeds, average predictions
Expected: 0.3-0.5% improvement (27.86% → 27.5%)
Runtime: +10 min
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

print("="*80)
print("ENSEMBLE: Multiple Seeds")
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
        label_encoders[col] = le

# Combine features
X_train = np.concatenate([train_df[tabular_features].values, train_pca], axis=1)
X_test = np.concatenate([test_df[tabular_features].values, test_pca], axis=1)
y_train = np.sqrt(train_df['price'].values)

# Train multiple models with different seeds
seeds = [42, 123, 456, 789, 2024]
all_predictions = []

for i, seed in enumerate(seeds, 1):
    print(f"\n[{i}/{len(seeds)}] Training model with seed={seed}...")
    
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
        'seed': seed,
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(params, train_data, num_boost_round=2000)
    
    pred_sqrt = model.predict(X_test)
    pred = np.maximum(pred_sqrt ** 2, 0)
    all_predictions.append(pred)
    
    print(f"  ✓ Seed {seed}: mean=${pred.mean():.2f}, median=${np.median(pred):.2f}")

# Ensemble: Average predictions
print("\n[Final] Averaging predictions...")
final_pred = np.mean(all_predictions, axis=0)

print(f"\nEnsemble stats:")
print(f"  Mean:   ${final_pred.mean():.2f}")
print(f"  Median: ${np.median(final_pred):.2f}")
print(f"  Range:  [${final_pred.min():.2f}, ${final_pred.max():.2f}]")

# Create submission
submission = pd.DataFrame({
    'sample_id': test_df['sample_id'],
    'price': final_pred
})
submission.to_csv('submission_ensemble_seeds.csv', index=False)
print(f"\n✅ Saved: submission_ensemble_seeds.csv")
