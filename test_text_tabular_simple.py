"""
QUICK TEST: Text + Tabular with Ridge Regression
=================================================
Fast check to see if combining text embeddings + tabular features
can beat the broken image embeddings (73.85% SMAPE)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("QUICK TEST: Text + Tabular (No Deep Learning)")
print("=" * 80)

# Load data
print("\n[1/4] Loading data...")
txt_emb = np.load('outputs/train_text_embeddings_clip.npy')
train_df = pd.read_csv('dataset/train_with_features.csv', low_memory=False)
y_train = train_df['price'].values
y_train_log = np.log1p(y_train)

print(f"✓ Text embeddings: {txt_emb.shape}")
print(f"✓ Prices: {len(y_train)}")

# Extract tabular features
print("\n[2/4] Extracting tabular features...")

def extract_tabular_features(df):
    features = pd.DataFrame()
    
    # Numerical
    for col in ['value', 'pack_size', 'text_length', 'word_count', 
                'num_bullets', 'description_length', 'price_per_unit']:
        if col in df.columns:
            features[col] = df[col].fillna(0)
    
    # Binary
    for col in ['has_premium', 'has_organic', 'has_gourmet', 'has_natural', 
                'has_artisan', 'has_luxury', 'is_travel_size', 'is_bulk', 
                'brand_exists', 'has_bullets']:
        if col in df.columns:
            features[col] = df[col].fillna(0).astype(int)
    
    # Interactions
    for col in ['value_x_premium', 'value_x_luxury', 'value_x_organic',
                'pack_x_premium', 'pack_x_value', 'brand_x_premium',
                'brand_x_organic', 'travel_x_value', 'bulk_x_value']:
        if col in df.columns:
            features[col] = df[col].fillna(0)
    
    # Categorical
    if 'unit' in df.columns:
        unit_map = {'kilogram': 0, 'gram': 1, 'litre': 2, 'millilitre': 3, 
                   'centimetre': 4, 'metre': 5, 'unit': 6, 'count': 7}
        features['unit_encoded'] = df['unit'].fillna('unit').map(unit_map).fillna(6)
    
    if 'price_segment' in df.columns:
        segment_map = {'budget': 0, 'economy': 1, 'mid_range': 2, 'premium': 3, 'luxury': 4}
        features['price_segment_encoded'] = df['price_segment'].fillna('mid_range').map(segment_map).fillna(2)
    
    if 'brand' in df.columns:
        brand_counts = df['brand'].value_counts()
        features['brand_freq'] = df['brand'].map(brand_counts).fillna(0) / len(df)
    
    return features.values

tabular = extract_tabular_features(train_df)
print(f"✓ Extracted {tabular.shape[1]} tabular features")

# Normalize
scaler = StandardScaler()
tabular_scaled = scaler.fit_transform(tabular)

# Split
indices = np.arange(len(y_train))
train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

X_txt_train, X_txt_val = txt_emb[train_idx], txt_emb[val_idx]
X_tab_train, X_tab_val = tabular_scaled[train_idx], tabular_scaled[val_idx]
y_log_train, y_log_val = y_train_log[train_idx], y_train_log[val_idx]
y_price_val = np.expm1(y_log_val)

# Test individual components
print("\n[3/4] Testing individual components...")

# Tabular only
model_tab = Ridge(alpha=1.0)
model_tab.fit(X_tab_train, y_log_train)
pred_tab = model_tab.predict(X_tab_val)
pred_tab_price = np.expm1(pred_tab)

r2_tab = r2_score(y_log_val, pred_tab)
smape_tab = 100 * np.mean(2 * np.abs(pred_tab_price - y_price_val) / 
                          (np.abs(pred_tab_price) + np.abs(y_price_val)))

print(f"\n📊 Tabular features only (Ridge):")
print(f"   R²: {r2_tab:.4f}")
print(f"   SMAPE: {smape_tab:.2f}%")
print(f"   Pred range: [${pred_tab_price.min():.2f}, ${pred_tab_price.max():.2f}]")

# Text only
model_txt = Ridge(alpha=1.0)
model_txt.fit(X_txt_train, y_log_train)
pred_txt = model_txt.predict(X_txt_val)
pred_txt_price = np.expm1(pred_txt)

r2_txt = r2_score(y_log_val, pred_txt)
smape_txt = 100 * np.mean(2 * np.abs(pred_txt_price - y_price_val) / 
                          (np.abs(pred_txt_price) + np.abs(y_price_val)))

print(f"\n📊 Text embeddings only (Ridge):")
print(f"   R²: {r2_txt:.4f}")
print(f"   SMAPE: {smape_txt:.2f}%")
print(f"   Pred range: [${pred_txt_price.min():.2f}, ${pred_txt_price.max():.2f}]")

# Combined
print("\n[4/4] Testing Text + Tabular combination...")

X_combined_train = np.concatenate([X_txt_train, X_tab_train], axis=1)
X_combined_val = np.concatenate([X_txt_val, X_tab_val], axis=1)

# Ridge
model_ridge = Ridge(alpha=1.0)
model_ridge.fit(X_combined_train, y_log_train)
pred_ridge = model_ridge.predict(X_combined_val)
pred_ridge_price = np.expm1(pred_ridge)

r2_ridge = r2_score(y_log_val, pred_ridge)
smape_ridge = 100 * np.mean(2 * np.abs(pred_ridge_price - y_price_val) / 
                            (np.abs(pred_ridge_price) + np.abs(y_price_val)))

print(f"\n📊 Text + Tabular (Ridge):")
print(f"   R²: {r2_ridge:.4f}")
print(f"   SMAPE: {smape_ridge:.2f}%")
print(f"   Pred range: [${pred_ridge_price.min():.2f}, ${pred_ridge_price.max():.2f}]")

# Random Forest (better for non-linear patterns)
print(f"\n   Training Random Forest (this may take a minute)...")
model_rf = RandomForestRegressor(n_estimators=100, max_depth=20, min_samples_split=10, 
                                 random_state=42, n_jobs=-1, verbose=0)
model_rf.fit(X_combined_train, y_log_train)
pred_rf = model_rf.predict(X_combined_val)
pred_rf_price = np.expm1(pred_rf)

r2_rf = r2_score(y_log_val, pred_rf)
smape_rf = 100 * np.mean(2 * np.abs(pred_rf_price - y_price_val) / 
                         (np.abs(pred_rf_price) + np.abs(y_price_val)))

print(f"\n📊 Text + Tabular (Random Forest):")
print(f"   R²: {r2_rf:.4f}")
print(f"   SMAPE: {smape_rf:.2f}%")
print(f"   Pred range: [${pred_rf_price.min():.2f}, ${pred_rf_price.max():.2f}]")

# Performance by price range
print(f"\n📈 Random Forest Performance by Price Range:")
for low, high in [(0, 10), (10, 25), (25, 50), (50, 100), (100, 1000), (1000, 10000)]:
    mask = (y_price_val >= low) & (y_price_val < high)
    if mask.sum() > 0:
        range_smape = 100 * np.mean(
            2 * np.abs(pred_rf_price[mask] - y_price_val[mask]) / 
            (np.abs(pred_rf_price[mask]) + np.abs(y_price_val[mask]))
        )
        print(f"   ${low:4d}-${high:5d}: {range_smape:6.2f}% SMAPE ({mask.sum():5d} samples)")

# Final summary
print("\n" + "=" * 80)
print("📊 SUMMARY: Text + Tabular vs Image MLP")
print("=" * 80)

print(f"\n🔴 Image MLP (BROKEN):         73.85% SMAPE")
print(f"   └─ R² = -0.0085 (no price signal)")

print(f"\n🟢 Text + Tabular (Ridge):     {smape_ridge:.2f}% SMAPE")
print(f"   ├─ R² = {r2_ridge:.4f}")
print(f"   └─ Improvement: {73.85 - smape_ridge:+.2f}%")

print(f"\n🟢 Text + Tabular (RF):        {smape_rf:.2f}% SMAPE")
print(f"   ├─ R² = {r2_rf:.4f}")
print(f"   └─ Improvement: {73.85 - smape_rf:+.2f}%")

print(f"\n💡 Key Findings:")
print(f"   1. Tabular features alone: {smape_tab:.2f}% SMAPE (R²={r2_tab:.4f})")
print(f"   2. Text embeddings alone:  {smape_txt:.2f}% SMAPE (R²={r2_txt:.4f})")
print(f"   3. Combined (best):        {smape_rf:.2f}% SMAPE (R²={r2_rf:.4f})")

print(f"\n🎯 Conclusion:")
if smape_rf < 45:
    print(f"   ✅ EXCELLENT! {smape_rf:.2f}% meets target (<45%)")
    print(f"   📤 With MLP, should easily hit 40-45% SMAPE")
    print(f"   🚀 PROCEED: Create Kaggle submission script")
elif smape_rf < 55:
    print(f"   ✅ GOOD! {smape_rf:.2f}% much better than image-based (73.85%)")
    print(f"   📈 With MLP, should reach ~{smape_rf-5:.0f}-{smape_rf:.0f}% SMAPE")
    print(f"   💡 PROCEED: Create Kaggle submission script")
elif smape_rf < 65:
    print(f"   ⚠️  MODERATE: {smape_rf:.2f}% better than images but below target")
    print(f"   💡 Consider: Feature engineering or ensemble methods")
else:
    print(f"   ❌ POOR: {smape_rf:.2f}% - investigate further")

print("=" * 80)
