"""
BASELINE MODEL - Incremental Feature Validation
================================================

PHILOSOPHY:
- Start with simplest model (IPQ only)
- Add features ONE GROUP at a time
- Measure impact of EACH addition
- Reason about why each feature should help

GOAL: Validate hypotheses from EDA, understand feature importance

Experiments:
1. IPQ only (value + unit) - validates if quantity pricing works
2. + Quality keywords - validates +46% premium hypothesis from EDA
3. + Brand - validates brand premium hypothesis
4. + Text features - validates description richness hypothesis

Each experiment outputs:
- 5-fold CV SMAPE
- Feature importance
- SMAPE by price segment (Budget/Mid/Premium/Luxury)
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import sys
import os
sys.path.append('.')
from config.config import N_SPLITS, RANDOM_STATE

# Use local path for baseline
DATA_PATH = 'dataset'


def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error"""
    denominator = (np.abs(y_true) + np.abs(y_pred))
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return 100 * np.mean(diff)  # Fixed: was 200, should be 100


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


def apply_segment_transform(prices, transform_type='adaptive'):
    """
    Apply segment-specific transforms to handle different price ranges
    
    adaptive: Budget=log, Mid-Range=none, Premium/Luxury=sqrt
    log: log transform for all
    sqrt: sqrt transform for all
    none: no transform
    """
    if transform_type == 'none':
        return prices, None
    
    transformed = np.zeros_like(prices)
    segment_masks = {
        'budget': prices < 10,
        'mid': (prices >= 10) & (prices < 50),
        'premium': (prices >= 50) & (prices < 100),
        'luxury': prices >= 100
    }
    
    if transform_type == 'adaptive':
        # Budget: log (compress low prices)
        transformed[segment_masks['budget']] = np.log1p(prices[segment_masks['budget']])
        # Mid-Range: no transform (already working!)
        transformed[segment_masks['mid']] = prices[segment_masks['mid']]
        # Premium: sqrt (less compression than log)
        transformed[segment_masks['premium']] = np.sqrt(prices[segment_masks['premium']])
        # Luxury: sqrt
        transformed[segment_masks['luxury']] = np.sqrt(prices[segment_masks['luxury']])
    elif transform_type == 'log':
        transformed = np.log1p(prices)
    elif transform_type == 'sqrt':
        transformed = np.sqrt(prices)
    
    return transformed, segment_masks


def inverse_segment_transform(transformed, original_prices, transform_type='adaptive'):
    """Inverse transform predictions back to original scale"""
    if transform_type == 'none':
        return transformed
    
    predictions = np.zeros_like(transformed)
    segment_masks = {
        'budget': original_prices < 10,
        'mid': (original_prices >= 10) & (original_prices < 50),
        'premium': (original_prices >= 50) & (original_prices < 100),
        'luxury': original_prices >= 100
    }
    
    if transform_type == 'adaptive':
        # Budget: exp
        predictions[segment_masks['budget']] = np.expm1(transformed[segment_masks['budget']])
        # Mid-Range: no transform
        predictions[segment_masks['mid']] = transformed[segment_masks['mid']]
        # Premium/Luxury: square
        predictions[segment_masks['premium']] = transformed[segment_masks['premium']] ** 2
        predictions[segment_masks['luxury']] = transformed[segment_masks['luxury']] ** 2
    elif transform_type == 'log':
        predictions = np.expm1(transformed)
    elif transform_type == 'sqrt':
        predictions = transformed ** 2
    
    return predictions


def train_lgb_cv(df, features, target='price', n_splits=5, transform='adaptive'):
    """
    Train LightGBM with K-Fold CV and segment-adaptive transforms
    
    Args:
    - transform: 'adaptive', 'log', 'sqrt', or 'none'
    
    Returns:
    - oof_predictions: Out-of-fold predictions for full dataset
    - feature_importance: Dict of feature importances
    - fold_scores: List of SMAPE per fold
    """
    # Encode categorical features
    cat_features = []
    for col in features:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = df[col].fillna('missing')
            df[col] = le.fit_transform(df[col].astype(str))
            cat_features.append(col)
    
    # Don't fill numeric NaN - LightGBM handles NaN natively
    
    X = df[features].values
    y_original = df[target].values
    
    # Apply segment-adaptive transform
    y, _ = apply_segment_transform(y_original, transform)
    
    # K-Fold CV
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    oof_predictions = np.zeros(len(df))
    feature_importance = {f: 0 for f in features}
    fold_scores = []
    
    print(f"\n{'='*60}")
    print(f"Training LightGBM with {len(features)} features")
    print(f"Features: {', '.join(features)}")
    print(f"{'='*60}\n")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # LightGBM params - MSE in log space approximates percentage error
        params = {
            'objective': 'regression',  # MSE on log-transformed targets
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'seed': RANDOM_STATE,
        }
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data, val_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=0)  # Suppress iteration logs
            ]
        )
        
        # Predict
        val_pred = model.predict(X_val)
        
        # Inverse transform predictions back to original scale
        y_val_original = y_original[val_idx]
        val_pred = inverse_segment_transform(val_pred, y_val_original, transform)
        
        oof_predictions[val_idx] = val_pred
        
        # Calculate SMAPE for this fold (using original prices)
        fold_smape = smape(y_val_original, val_pred)
        fold_scores.append(fold_smape)
        print(f"Fold {fold}: {fold_smape:.2f}%", end="  ")
        
        # Accumulate feature importance
        for i, feat in enumerate(features):
            feature_importance[feat] += model.feature_importance()[i]
    
    print()  # Newline after all folds
    
    # Average feature importance across folds
    feature_importance = {k: v/n_splits for k, v in feature_importance.items()}
    
    return oof_predictions, feature_importance, fold_scores


def print_results(df, oof_preds, feature_importance, fold_scores, experiment_name):
    """Print experiment results - COMPACT FORMAT"""
    print(f"\n{'='*70}")
    print(f"EXP: {experiment_name}")
    print(f"{'='*70}")
    
    # Overall SMAPE
    overall_smape = smape(df['price'].values, oof_preds)
    print(f"OOF SMAPE: {overall_smape:.2f}%  |  Folds: {[f'{s:.2f}' for s in fold_scores]}  |  Std: {np.std(fold_scores):.2f}%")
    
    # SMAPE by segment
    df_eval = df.copy()
    df_eval['pred'] = oof_preds
    segment_smapes = smape_by_segment(df_eval)
    
    print("\nBy Segment:", end="  ")
    for segment, seg_smape in segment_smapes.items():
        count = (df['price_segment'] == segment).sum()
        pct = count / len(df) * 100
        print(f"{segment}: {seg_smape:.2f}% ({pct:.0f}%)", end="  ")
    
    # Feature importance - Top 5 only
    print(f"\n\nTop 5 Features:", end="  ")
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    for feat, imp in sorted_features[:5]:
        print(f"{feat}({imp:.0f})", end="  ")
    
    print(f"\n{'='*70}\n")
    
    return overall_smape, segment_smapes


# ============================================================================
# EXPERIMENT 1: IPQ ONLY (value + unit)
# ============================================================================
# REASONING:
# - EDA showed IPQ has 84.8% coverage (63,600 / 75,000 products)
# - Items without IPQ have NaN values (-999 after filling)
# - Hypothesis: Larger quantities = lower per-unit price
# - Expected SMAPE: 15-18% (baseline)
# ============================================================================

def experiment_1_ipq_only():
    """Baseline: IPQ only"""
    print("\n" + "="*70)
    print("EXP 1: IPQ ONLY (value + unit) - Expected: 16-19% SMAPE")
    print("="*70)
    
    df = pd.read_csv(f'{DATA_PATH}/train_with_features.csv', low_memory=False)
    
    features = ['value', 'unit']  # IPQ = Item-Pack-Quantity
    
    oof_preds, feat_imp, fold_scores = train_lgb_cv(df, features)
    overall_smape, segment_smapes = print_results(
        df, oof_preds, feat_imp, fold_scores,
        "IPQ Only (value+unit)"
    )
    
    return overall_smape


# ============================================================================
# EXPERIMENT 2: IPQ + Quality Keywords
# ============================================================================
# REASONING:
# - EDA showed quality keywords → +46% price premium
# - Premium: 18% coverage, Organic: 12%, Gourmet: 8%, etc
# - These are BINARY signals (has keyword or not)
# - Hypothesis: Quality keywords segment products into price tiers
# - Expected improvement: 15-18% → 13-16% SMAPE (2-3% reduction)
# ============================================================================

def experiment_2_ipq_plus_quality():
    """IPQ + Quality keywords"""
    print("\n" + "="*70)
    print("EXP 2: IPQ + QUALITY - Expected: 14-17% SMAPE")
    print("="*70)
    
    df = pd.read_csv(f'{DATA_PATH}/train_with_features.csv', low_memory=False)
    
    features = [
        'value', 'unit',  # IPQ = Item-Pack-Quantity
        'has_premium', 'has_organic', 'has_gourmet',
        'has_natural', 'has_artisan', 'has_luxury'
    ]
    
    oof_preds, feat_imp, fold_scores = train_lgb_cv(df, features)
    overall_smape, segment_smapes = print_results(
        df, oof_preds, feat_imp, fold_scores,
        "IPQ + Quality Keywords"
    )
    
    return overall_smape


# ============================================================================
# EXPERIMENT 3: IPQ + Quality + Brand
# ============================================================================
# REASONING:
# - Brand extracted for 77.1% of products (57,846 / 75,000)
# - Different brands → different pricing strategies
# - Examples: "Log Cabin" (established), "Walden Farms" (premium sugar-free)
# - Hypothesis: Brand captures manufacturer price positioning
# - Challenge: High cardinality (many unique brands)
# - Expected improvement: 1-2% SMAPE reduction
# ============================================================================

def experiment_3_ipq_quality_brand():
    """IPQ + Quality + Brand"""
    print("\n" + "="*70)
    print("EXP 3: IPQ + QUALITY + BRAND - Expected: 13-15% SMAPE")
    print("="*70)
    
    df = pd.read_csv(f'{DATA_PATH}/train_with_features.csv', low_memory=False)
    
    features = [
        'value', 'unit',  # IPQ = Item-Pack-Quantity
        'has_premium', 'has_organic', 'has_gourmet',
        'has_natural', 'has_artisan', 'has_luxury',
        'brand'
    ]
    
    oof_preds, feat_imp, fold_scores = train_lgb_cv(df, features)
    overall_smape, segment_smapes = print_results(
        df, oof_preds, feat_imp, fold_scores,
        "IPQ + Quality + Brand"
    )
    
    return overall_smape


# ============================================================================
# EXPERIMENT 4: IPQ + QUALITY + BRAND + PACK SIZE
# ============================================================================
# REASONING:
# - Add pack_size: bulk purchases → lower per-unit price
# - Removed weak text features (text_length r=0.14, added noise in run #4)
# - Hypothesis: Pack size captures bulk pricing without description noise
# - Expected improvement: 0.5-1% SMAPE reduction over Experiment 3
# ============================================================================

def experiment_4_full_tabular():
    """IPQ + Quality + Brand + Pack + Size Signals + Interactions"""
    print("\n" + "="*70)
    print("EXP 4: FULL + INTERACTIONS (adaptive transform)")
    print("="*70)
    
    df = pd.read_csv(f'{DATA_PATH}/train_with_features.csv', low_memory=False)
    
    features = [
        'value', 'unit', 'pack_size',  # IPQ + bulk indicator
        'has_premium', 'has_organic', 'has_gourmet',
        'has_natural', 'has_artisan', 'has_luxury',
        'brand',
        'is_travel_size', 'is_bulk',  # Size category signals
        'value_x_premium', 'value_x_luxury', 'value_x_organic',  # Interactions
        'pack_x_premium', 'pack_x_value'
    ]
    
    oof_preds, feat_imp, fold_scores = train_lgb_cv(df, features, transform='adaptive')
    overall_smape, segment_smapes = print_results(
        df, oof_preds, feat_imp, fold_scores,
        "Full + Interactions (Adaptive)"
    )
    
    return overall_smape


# ============================================================================
# MAIN: RUN ALL EXPERIMENTS
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("BASELINE #5: IPQ + Quality + Brand + Size Signals")
    print("="*70 + "\n")
    
    results = {}
    
    # Run experiments
    results['ipq_only'] = experiment_1_ipq_only()
    results['ipq_quality'] = experiment_2_ipq_plus_quality()
    results['ipq_quality_brand'] = experiment_3_ipq_quality_brand()
    results['full_tabular'] = experiment_4_full_tabular()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    baseline = results['ipq_only']
    print(f"Exp 1 (IPQ):          {baseline:.2f}% (baseline)")
    
    for name, smape in list(results.items())[1:]:
        improvement = baseline - smape
        improvement_pct = (improvement / baseline) * 100
        display_name = name.replace('_', ' ').title()
        print(f"{display_name:<20}: {smape:>5.2f}%  ({improvement:>+4.2f}% / {improvement_pct:>+4.1f}%)")
    
    best_smape = min(results.values())
    print(f"\n✨ Best: {best_smape:.2f}% SMAPE")
    
    if best_smape < 13:
        print("→ STRONG! Try TF-IDF on bullets next")
    elif best_smape < 15:
        print("→ GOOD. Try TF-IDF or tune hyperparams")
    else:
        print("→ DEBUG: Check categorical encoding, unit cardinality")
    
    print("="*70 + "\n")
    print("4. Decide: Add image embeddings? (60-90 min)")
    print("5. Optimize hyperparameters if needed")
