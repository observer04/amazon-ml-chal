"""
Reproduce the exact 27.86% validation score to understand what's different
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

RANDOM_STATE = 42

print("REPRODUCING BASELINE 27.86% VALIDATION")
print("="*80)

# Load data
train_df = pd.read_csv('dataset/train_with_features.csv', low_memory=False)
train_txt = np.load('outputs/train_text_embeddings_clip.npy')

# PCA
pca = PCA(n_components=20, random_state=RANDOM_STATE)
train_pca = pca.fit_transform(train_txt)
print(f"PCA: {pca.explained_variance_ratio_.sum():.2%} variance")

# Features (EXACT same as baseline)
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
df_encoded = train_df.copy()
for col in tabular_features:
    if df_encoded[col].dtype == 'object':
        le = LabelEncoder()
        df_encoded[col] = df_encoded[col].fillna('missing')
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    else:
        df_encoded[col] = df_encoded[col].fillna(df_encoded[col].median())

X_tabular = df_encoded[tabular_features].values
X_combined = np.concatenate([X_tabular, train_pca], axis=1)
y = np.sqrt(train_df['price'].values)  # SQRT transform

print(f"Features: {X_combined.shape}")
print(f"Target: SQRT(price)")

# 5-fold CV (EXACT same as baseline)
kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
oof_predictions = np.zeros(len(train_df))
fold_scores = []

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

print("\n5-Fold CV:")
for fold, (train_idx, val_idx) in enumerate(kf.split(X_combined), 1):
    X_train, X_val = X_combined[train_idx], X_combined[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=2000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
    )
    
    # Predict
    val_pred_sqrt = model.predict(X_val)
    val_pred = np.maximum(val_pred_sqrt ** 2, 0)
    
    # Get original prices
    y_val_orig = train_df['price'].values[val_idx]
    
    # Calculate SMAPE
    score = smape(y_val_orig, val_pred)
    fold_scores.append(score)
    oof_predictions[val_idx] = val_pred
    
    print(f"  Fold {fold}: {score:.2f}%")

print(f"\nOverall: {np.mean(fold_scores):.2f}% ± {np.std(fold_scores):.2f}%")
print(f"Expected: 27.86% (from baseline_enhanced.py)")

if abs(np.mean(fold_scores) - 27.86) < 1:
    print("\n✓ REPRODUCED! Scores match")
else:
    print(f"\n✗ MISMATCH: {np.mean(fold_scores):.2f}% vs 27.86%")
    print("   Investigating difference...")
