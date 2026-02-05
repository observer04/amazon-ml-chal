"""
ENHANCED BASELINE: Tabular + Text Embedding Features
======================================================

STRATEGY:
- Keep your proven 30.70% tabular baseline (22 features)
- Add top 20 PCA components from text CLIP embeddings (R²=0.33)
- Expected: 28-30% SMAPE (2-3% improvement)

REASONING:
- Text embeddings capture semantic signals (R²=0.33)
- PCA reduces 512 dims → 20 dims (removes noise, keeps signal)
- Combined with tabular features for synergy
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import sys
import os
sys.path.append('.')
from config.config import N_SPLITS, RANDOM_STATE

DATA_PATH = 'dataset'


def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error"""
    denominator = (np.abs(y_true) + np.abs(y_pred))
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return 100 * np.mean(diff)


def smape_by_segment(df, y_true_col='price', y_pred_col='pred', segment_col='price_segment'):
    """Calculate SMAPE for each price segment"""
    results = {}
    for segment in df[segment_col].unique():
        mask = df[segment_col] == segment
        if mask.sum() == 0:
            continue
        segment_smape = smape(
            df.loc[mask, y_true_col].values,
            df.loc[mask, y_pred_col].values
        )
        results[segment] = segment_smape
    return results


def extract_text_pca_features(n_components=20):
    """
    Extract PCA features from text embeddings
    
    Returns:
    - pca_features: (75000, n_components) array
    - pca: fitted PCA object (for test set transform)
    """
    print(f"\n{'='*70}")
    print(f"Extracting {n_components} PCA components from text embeddings...")
    print(f"{'='*70}")
    
    # Load text embeddings
    txt_emb = np.load('outputs/train_text_embeddings_clip.npy')
    print(f"✓ Loaded text embeddings: {txt_emb.shape}")
    
    # Fit PCA
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    pca_features = pca.fit_transform(txt_emb)
    
    # Show explained variance
    explained_var = pca.explained_variance_ratio_.sum()
    print(f"✓ PCA fitted: {n_components} components")
    print(f"✓ Explained variance: {explained_var:.2%}")
    print(f"✓ Top 5 components: {pca.explained_variance_ratio_[:5]}")
    
    return pca_features, pca


def train_lgb_cv_enhanced(df, tabular_features, text_pca_features, target='price', n_splits=5):
    """
    Train LightGBM with tabular + text PCA features
    """
    # Encode categorical features
    df_encoded = df.copy()
    cat_features = []
    for col in tabular_features:
        if df_encoded[col].dtype == 'object':
            le = LabelEncoder()
            df_encoded[col] = df_encoded[col].fillna('missing')
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            cat_features.append(col)
    
    # Combine tabular + text PCA
    X_tabular = df_encoded[tabular_features].values
    X_combined = np.concatenate([X_tabular, text_pca_features], axis=1)
    
    # Create feature names
    feature_names = tabular_features + [f'text_pca_{i}' for i in range(text_pca_features.shape[1])]
    
    y = np.sqrt(df[target].values)  # SQRT transform
    
    # K-Fold CV
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    oof_predictions = np.zeros(len(df))
    feature_importance = {f: 0 for f in feature_names}
    fold_scores = []
    
    print(f"\n{'='*70}")
    print(f"Training LightGBM with {len(feature_names)} features")
    print(f"  - Tabular: {len(tabular_features)}")
    print(f"  - Text PCA: {text_pca_features.shape[1]}")
    print(f"{'='*70}\n")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_combined), 1):
        X_train, X_val = X_combined[train_idx], X_combined[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # LightGBM params - same as baseline
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
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=2000,
            valid_sets=[train_data, val_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=0)
            ]
        )
        
        # Predict and inverse transform
        val_pred_sqrt = model.predict(X_val)
        val_pred = np.maximum(val_pred_sqrt ** 2, 0)
        
        oof_predictions[val_idx] = val_pred
        
        # Calculate SMAPE
        y_val_original = df.iloc[val_idx][target].values
        fold_smape = smape(y_val_original, val_pred)
        fold_scores.append(fold_smape)
        print(f"Fold {fold}: {fold_smape:.2f}%", end="  ")
        
        # Accumulate feature importance
        for i, feat in enumerate(feature_names):
            feature_importance[feat] += model.feature_importance()[i]
    
    print()
    
    # Average feature importance
    feature_importance = {k: v/n_splits for k, v in feature_importance.items()}
    
    return oof_predictions, feature_importance, fold_scores, feature_names


def print_results(df, oof_preds, feature_importance, fold_scores, feature_names, n_text_features):
    """Print experiment results"""
    print(f"\n{'='*70}")
    print(f"ENHANCED MODEL: Tabular + Text PCA")
    print(f"{'='*70}")
    
    # Overall SMAPE
    overall_smape = smape(df['price'].values, oof_preds)
    print(f"OOF SMAPE: {overall_smape:.2f}%")
    print(f"Folds: {[f'{s:.2f}' for s in fold_scores]}")
    print(f"Std: {np.std(fold_scores):.2f}%")
    
    # SMAPE by segment
    df_eval = df.copy()
    df_eval['pred'] = oof_preds
    segment_smapes = smape_by_segment(df_eval)
    
    print(f"\n{'='*70}")
    print("By Segment:")
    for segment, seg_smape in segment_smapes.items():
        count = (df['price_segment'] == segment).sum()
        pct = count / len(df) * 100
        print(f"  {segment:12s}: {seg_smape:6.2f}% SMAPE ({count:5d} items, {pct:4.1f}%)")
    
    # Feature importance - Top 10
    print(f"\n{'='*70}")
    print("Top 10 Features:")
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    for i, (feat, imp) in enumerate(sorted_features[:10], 1):
        is_text = 'text_pca' in feat
        marker = '📝' if is_text else '📊'
        print(f"  {i:2d}. {marker} {feat:20s}: {imp:8.0f}")
    
    # Count text features in top 10
    text_in_top10 = sum(1 for feat, _ in sorted_features[:10] if 'text_pca' in feat)
    print(f"\n  → {text_in_top10}/10 top features are text PCA")
    
    # Importance by category
    tabular_importance = sum(imp for feat, imp in feature_importance.items() if 'text_pca' not in feat)
    text_importance = sum(imp for feat, imp in feature_importance.items() if 'text_pca' in feat)
    total_importance = tabular_importance + text_importance
    
    print(f"\n{'='*70}")
    print("Importance by Category:")
    print(f"  Tabular features: {tabular_importance/total_importance*100:5.1f}% ({len(feature_names) - n_text_features} features)")
    print(f"  Text PCA:         {text_importance/total_importance*100:5.1f}% ({n_text_features} features)")
    
    print(f"{'='*70}\n")
    
    return overall_smape, segment_smapes


# ============================================================================
# EXPERIMENT: BASELINE VS ENHANCED
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("ENHANCED BASELINE: Tabular + Text Embeddings")
    print("="*70 + "\n")
    
    # Load data
    df = pd.read_csv(f'{DATA_PATH}/train_with_features.csv', low_memory=False)
    
    # Define tabular features (same as baseline)
    tabular_features = [
        # Core IPQ
        'value', 'unit', 'pack_size',
        
        # Quality signals
        'has_premium', 'has_organic', 'has_gourmet',
        'has_natural', 'has_artisan', 'has_luxury',
        
        # Size categories
        'is_travel_size', 'is_bulk',
        
        # Brand
        'brand', 'brand_exists',
        
        # Interactions - Value×Quality
        'value_x_premium', 'value_x_luxury', 'value_x_organic',
        
        # Interactions - Pack×Quality
        'pack_x_premium', 'pack_x_value',
        
        # Interactions - Brand×Quality
        'brand_x_premium', 'brand_x_organic',
        
        # Interactions - Size×Value
        'travel_x_value', 'bulk_x_value'
    ]
    
    # Extract text PCA features
    n_pca_components = 20
    text_pca_features, pca_model = extract_text_pca_features(n_components=n_pca_components)
    
    # Train enhanced model
    oof_preds, feat_imp, fold_scores, feature_names = train_lgb_cv_enhanced(
        df, tabular_features, text_pca_features
    )
    
    enhanced_smape, segment_smapes = print_results(
        df, oof_preds, feat_imp, fold_scores, feature_names, n_pca_components
    )
    
    # Comparison with baseline
    baseline_smape = 30.70  # From your baseline run
    
    print("\n" + "="*70)
    print("COMPARISON: Baseline vs Enhanced")
    print("="*70)
    print(f"Baseline (Tabular only):       {baseline_smape:.2f}% SMAPE")
    print(f"Enhanced (Tabular + Text PCA): {enhanced_smape:.2f}% SMAPE")
    print(f"Improvement:                   {baseline_smape - enhanced_smape:+.2f}%")
    print(f"Relative improvement:          {(baseline_smape - enhanced_smape)/baseline_smape*100:+.1f}%")
    
    print(f"\n{'='*70}")
    print("VERDICT:")
    print(f"{'='*70}")
    
    if enhanced_smape < 28:
        print("✅✅ EXCELLENT! >2.5% improvement")
        print("   → Ready for final submission")
    elif enhanced_smape < 29:
        print("✅ GOOD! 1.5-2.5% improvement")
        print("   → Consider ensemble for additional boost")
    elif enhanced_smape < 30:
        print("✅ MODERATE: 0.5-1.5% improvement")
        print("   → Proceed with submission or try ensemble")
    elif enhanced_smape < baseline_smape:
        print("⚠️  SMALL improvement (<0.5%)")
        print("   → Text PCA helps slightly, can use baseline instead")
    else:
        print("❌ NO IMPROVEMENT - stick with baseline")
        print("   → Text embeddings don't help your tabular features")
    
    print(f"\n{'='*70}")
    print("NEXT STEPS:")
    print(f"{'='*70}")
    print("1. Save PCA model for test set transformation")
    print("2. Generate test predictions")
    print("3. Create Kaggle submission")
    print(f"{'='*70}\n")
    
    # Save PCA model
    import pickle
    with open('outputs/text_pca_model.pkl', 'wb') as f:
        pickle.dump(pca_model, f)
    print("✓ Saved PCA model to: outputs/text_pca_model.pkl")
