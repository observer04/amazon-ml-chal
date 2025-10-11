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


def train_lgb_cv(df, features, target='price', n_splits=5):
    """
    Train LightGBM with K-Fold CV
    
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
    # df[features] = df[features].fillna(-999)  # REMOVED: Let LightGBM handle missing values
    
    X = df[features].values
    y = df[target].values
    
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
        
        # LightGBM params - CONSERVATIVE for baseline
        params = {
            'objective': 'regression',
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
                lgb.log_evaluation(period=100)
            ]
        )
        
        # Predict
        val_pred = model.predict(X_val)
        oof_predictions[val_idx] = val_pred
        
        # Calculate SMAPE for this fold
        fold_smape = smape(y_val, val_pred)
        fold_scores.append(fold_smape)
        
        print(f"\nFold {fold} SMAPE: {fold_smape:.4f}%")
        
        # Accumulate feature importance
        for i, feat in enumerate(features):
            feature_importance[feat] += model.feature_importance()[i]
    
    # Average feature importance across folds
    feature_importance = {k: v/n_splits for k, v in feature_importance.items()}
    
    return oof_predictions, feature_importance, fold_scores


def print_results(df, oof_preds, feature_importance, fold_scores, experiment_name):
    """Print experiment results with reasoning"""
    print(f"\n{'='*80}")
    print(f"EXPERIMENT: {experiment_name}")
    print(f"{'='*80}\n")
    
    # Overall SMAPE
    overall_smape = smape(df['price'].values, oof_preds)
    print(f"Overall OOF SMAPE: {overall_smape:.4f}%")
    print(f"Fold scores: {[f'{s:.4f}%' for s in fold_scores]}")
    print(f"Std dev: {np.std(fold_scores):.4f}%\n")
    
    # SMAPE by segment
    df_eval = df.copy()
    df_eval['pred'] = oof_preds
    segment_smapes = smape_by_segment(df_eval)
    
    print("SMAPE by Price Segment:")
    print("-" * 50)
    for segment, seg_smape in segment_smapes.items():
        count = (df['price_segment'] == segment).sum()
        pct = count / len(df) * 100
        print(f"  {segment:12s}: {seg_smape:6.2f}%  (n={count:,}, {pct:.1f}% of data)")
    
    # Feature importance
    print("\nTop 10 Feature Importances:")
    print("-" * 50)
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    for i, (feat, imp) in enumerate(sorted_features[:10], 1):
        print(f"  {i:2d}. {feat:20s}: {imp:8.1f}")
    
    print(f"\n{'='*80}\n")
    
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
    print("\n" + "="*80)
    print("EXPERIMENT 1: IPQ ONLY (value + unit)")
    print("="*80)
    print("\nREASONING:")
    print("- EDA showed value/unit in 84.8% of products (63,600 / 75,000)")
    print("- Quantity is a strong price signal (12oz vs 48oz)")
    print("- This validates if IPQ alone is sufficient")
    print("- Expected: 15-18% SMAPE (baseline to beat)")
    print("\n")
    
    df = pd.read_csv(f'{DATA_PATH}/train_with_features.csv')
    
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
    print("\n" + "="*80)
    print("EXPERIMENT 2: IPQ + QUALITY KEYWORDS")
    print("="*80)
    print("\nREASONING:")
    print("- EDA showed products with quality keywords → +46% price premium")
    print("- Coverage: Premium 18%, Organic 12%, Gourmet 8%, Natural 7%, etc")
    print("- Binary flags (0/1) easy for tree model to split on")
    print("- Hypothesis: Keywords segment products into price tiers")
    print("- Expected improvement: 2-3% SMAPE reduction")
    print("\n")
    
    df = pd.read_csv(f'{DATA_PATH}/train_with_features.csv')
    
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
    print("\n" + "="*80)
    print("EXPERIMENT 3: IPQ + QUALITY + BRAND")
    print("="*80)
    print("\nREASONING:")
    print("- Brand extracted for 77.1% of products (57,846 / 75,000)")
    print("- Different brands have different pricing strategies")
    print("- Examples: Premium brands (Walden Farms), Budget brands")
    print("- Hypothesis: Brand captures manufacturer price positioning")
    print("- Challenge: High cardinality (many unique brands)")
    print("- Expected improvement: 1-2% SMAPE reduction")
    print("\n")
    
    df = pd.read_csv(f'{DATA_PATH}/train_with_features.csv')
    
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
    """IPQ + Quality + Brand + Pack Size"""
    print("\n" + "="*80)
    print("EXPERIMENT 4: IPQ + QUALITY + BRAND + PACK SIZE")
    print("="*80)
    print("\nREASONING:")
    print("- Add pack_size: bulk purchases → lower per-unit price")
    print("- Removed weak text features (r=0.14, added noise)")
    print("- Hypothesis: Pack size captures bulk pricing patterns")
    print("- Expected improvement: 0.5-1% SMAPE reduction")
    print("\n")
    
    df = pd.read_csv(f'{DATA_PATH}/train_with_features.csv')
    
    features = [
        'value', 'unit', 'pack_size',  # IPQ + bulk indicator
        'has_premium', 'has_organic', 'has_gourmet',
        'has_natural', 'has_artisan', 'has_luxury',
        'brand'
        # Removed text features: text_length, word_count, has_bullets, num_bullets
        # Reason: Weak correlation (r=0.14), added noise in previous run
    ]
    
    oof_preds, feat_imp, fold_scores = train_lgb_cv(df, features)
    overall_smape, segment_smapes = print_results(
        df, oof_preds, feat_imp, fold_scores,
        "IPQ + Quality + Brand + Pack"
    )
    
    return overall_smape


# ============================================================================
# MAIN: RUN ALL EXPERIMENTS
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*80)
    print("BASELINE MODEL - INCREMENTAL FEATURE VALIDATION")
    print("="*80)
    print("\nObjective: Validate hypotheses from EDA")
    print("Approach: Start simple, add features incrementally")
    print("Metric: SMAPE (lower is better)")
    print("\n")
    
    results = {}
    
    # Run experiments
    results['ipq_only'] = experiment_1_ipq_only()
    results['ipq_quality'] = experiment_2_ipq_plus_quality()
    results['ipq_quality_brand'] = experiment_3_ipq_quality_brand()
    results['full_tabular'] = experiment_4_full_tabular()
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY OF ALL EXPERIMENTS")
    print("="*80)
    print(f"\n{'Experiment':<30} {'SMAPE':<10} {'Improvement':<15}")
    print("-" * 60)
    
    baseline = results['ipq_only']
    print(f"{'1. IPQ Only':<30} {baseline:>6.2f}%  {'(baseline)':<15}")
    
    for name, smape in list(results.items())[1:]:
        improvement = baseline - smape
        improvement_pct = (improvement / baseline) * 100
        display_name = name.replace('_', ' ').title()
        print(f"{display_name:<30} {smape:>6.2f}%  {improvement:>+5.2f}% ({improvement_pct:>+4.1f}%)")
    
    print("\n" + "="*80)
    print("\nKEY INSIGHTS:")
    print("1. Which features contribute most to prediction?")
    print("2. Is Budget segment (<$10) the main error contributor?")
    print("3. Is IPQ alone sufficient, or do we need text/image embeddings?")
    print("4. What's the best SMAPE we can achieve with simple features?")
    print("\n")
    
    # Decision point
    best_smape = min(results.values())
    print(f"Best baseline SMAPE: {best_smape:.2f}%")
    print("\nDECISION TREE:")
    if best_smape < 13:
        print("✅ Strong baseline (<13%) → Can try text/image embeddings for marginal gains")
    elif best_smape < 15:
        print("⚠️ Moderate baseline (13-15%) → Text embeddings may help, reconsider image cost")
    else:
        print("🔴 Weak baseline (>15%) → Debug features, check for data leakage, tune hyperparams")
    
    print("\nNEXT STEPS:")
    print("1. Analyze feature importance → Which features matter most?")
    print("2. Check Budget segment errors → Is it the main contributor?")
    print("3. Decide: Add text embeddings? (30-40 min)")
    print("4. Decide: Add image embeddings? (60-90 min)")
    print("5. Optimize hyperparameters if needed")
