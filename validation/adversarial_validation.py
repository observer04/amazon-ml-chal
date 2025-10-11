"""
Adversarial Validation - Train vs Test Distribution Check
=========================================================

Purpose: Detect if train and test datasets come from different distributions.

Method:
1. Combine train + test datasets
2. Label them: train=0, test=1
3. Train a classifier to distinguish them
4. If classifier achieves >60% accuracy → distributions differ!
5. Feature importance shows which features differ most

Why this matters:
- If train ≠ test, models may not generalize
- Identifies risky features
- Helps explain 62% LB vs 30% CV gap
- Guides feature selection

Expected findings:
- If AUC > 0.6: distributions differ (problem!)
- If AUC ~ 0.5: distributions similar (good!)
- Feature importance: which features leaked
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')


def adversarial_validation(train_features, test_features, feature_cols):
    """
    Check if train and test come from same distribution.
    
    Returns:
        auc_score: Cross-validated AUC
        feature_importance: DataFrame of feature importance
    """
    print(f"\n🔬 Running Adversarial Validation...")
    print(f"   Train samples: {len(train_features):,}")
    print(f"   Test samples: {len(test_features):,}")
    print(f"   Features used: {len(feature_cols)}")
    
    # Add labels
    train_features['is_test'] = 0
    test_features['is_test'] = 1
    
    # Combine
    combined = pd.concat([train_features, test_features], axis=0, ignore_index=True)
    
    print(f"   Combined shape: {combined.shape}")
    
    # Prepare data
    X = combined[feature_cols].copy()
    y = combined['is_test']
    
    # Identify categorical columns
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    # Convert categorical to codes
    for col in cat_cols:
        X[col] = X[col].astype('category').cat.codes
    
    # Handle missing values (fill with -999 for LightGBM)
    X = X.fillna(-999)
    
    # Train classifier
    print(f"\n🤖 Training classifier to distinguish train vs test...")
    
    clf = LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        num_leaves=31,
        random_state=42,
        verbose=-1
    )
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
    
    mean_auc = cv_scores.mean()
    std_auc = cv_scores.std()
    
    print(f"\n📊 Cross-Validation AUC Scores:")
    for i, score in enumerate(cv_scores, 1):
        print(f"   Fold {i}: {score:.4f}")
    
    print(f"\n{'='*60}")
    print(f"Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"{'='*60}")
    
    # Interpretation
    print(f"\n🎯 Interpretation:")
    if mean_auc < 0.55:
        print(f"   ✅ EXCELLENT: Train and test are very similar!")
        print(f"   → Models should generalize well")
        risk_level = "LOW"
    elif mean_auc < 0.60:
        print(f"   🟡 GOOD: Slight difference, but acceptable")
        print(f"   → Some distribution shift, but manageable")
        risk_level = "MEDIUM"
    elif mean_auc < 0.70:
        print(f"   ⚠️  WARNING: Significant distribution difference!")
        print(f"   → Models may not generalize well")
        print(f"   → Some features differ between train/test")
        risk_level = "HIGH"
    else:
        print(f"   🔴 CRITICAL: Train and test are very different!")
        print(f"   → High risk of poor generalization")
        print(f"   → Major feature distribution shifts")
        risk_level = "CRITICAL"
    
    # Train full model for feature importance
    print(f"\n🔍 Analyzing feature importance...")
    clf.fit(X, y)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n📊 Top 15 Features Distinguishing Train vs Test:")
    print(f"{'Feature':<25} {'Importance':>12} {'Risk'}")
    print(f"{'-'*50}")
    
    for idx, row in feature_importance.head(15).iterrows():
        feat = row['feature']
        imp = row['importance']
        
        # Risk assessment
        if imp > 100:
            risk = "🔴 HIGH"
        elif imp > 50:
            risk = "⚠️  MEDIUM"
        else:
            risk = "🟢 LOW"
        
        print(f"{feat:<25} {imp:>12.0f} {risk}")
    
    return mean_auc, std_auc, feature_importance, risk_level


def check_feature_coverage(train_df, test_df, feature_cols):
    """Compare feature coverage between train and test."""
    print(f"\n\n{'='*60}")
    print(f"FEATURE COVERAGE COMPARISON")
    print(f"{'='*60}")
    
    coverage_diff = []
    
    print(f"\n{'Feature':<25} {'Train':>10} {'Test':>10} {'Diff':>10} {'Status'}")
    print(f"{'-'*70}")
    
    for feat in feature_cols:
        train_cov = train_df[feat].notna().mean()
        test_cov = test_df[feat].notna().mean()
        diff = abs(train_cov - test_cov)
        
        coverage_diff.append({
            'feature': feat,
            'train_coverage': train_cov,
            'test_coverage': test_cov,
            'difference': diff
        })
        
        # Status
        if diff < 0.05:
            status = "✅ OK"
        elif diff < 0.10:
            status = "🟡 WATCH"
        else:
            status = "⚠️  RISKY"
        
        print(f"{feat:<25} {train_cov:>9.1%} {test_cov:>9.1%} {diff:>9.1%} {status}")
    
    coverage_df = pd.DataFrame(coverage_diff).sort_values('difference', ascending=False)
    
    # Summary
    print(f"\n📊 Summary:")
    risky = (coverage_df['difference'] > 0.10).sum()
    watch = ((coverage_df['difference'] >= 0.05) & (coverage_df['difference'] <= 0.10)).sum()
    ok = (coverage_df['difference'] < 0.05).sum()
    
    print(f"   ⚠️  Risky features (>10% diff): {risky}")
    print(f"   🟡 Watch features (5-10% diff): {watch}")
    print(f"   ✅ Safe features (<5% diff): {ok}")
    
    return coverage_df


def main():
    """Run adversarial validation."""
    print("="*60)
    print("ADVERSARIAL VALIDATION: Train vs Test Distribution")
    print("="*60)
    
    # Load data
    print("\n📂 Loading data...")
    train_df = pd.read_csv('dataset/train_with_features.csv')
    test_df = pd.read_csv('dataset/test_with_features.csv')
    
    print(f"✅ Train: {train_df.shape}")
    print(f"✅ Test: {test_df.shape}")
    
    # Select features for validation (exclude IDs, text, price)
    exclude_cols = ['sample_id', 'catalog_content', 'image_link', 'price', 
                   'item_name', 'description', 'all_bullets_text',
                   'bullet_1', 'bullet_2', 'bullet_3', 'bullet_4', 'bullet_5', 'bullet_6',
                   'price_per_unit', 'price_segment']
    
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]
    
    print(f"\n✅ Selected {len(feature_cols)} features for validation:")
    print(f"   {', '.join(feature_cols[:10])}...")
    
    # Prepare datasets
    train_features = train_df[feature_cols + ['sample_id']].copy()
    test_features = test_df[feature_cols + ['sample_id']].copy()
    
    # Run adversarial validation
    mean_auc, std_auc, feature_importance, risk_level = adversarial_validation(
        train_features, 
        test_features, 
        feature_cols
    )
    
    # Check feature coverage
    coverage_df = check_feature_coverage(train_df, test_df, feature_cols)
    
    # Recommendations
    print(f"\n\n{'='*60}")
    print(f"RECOMMENDATIONS")
    print(f"{'='*60}")
    
    print(f"\n🎯 Overall Risk Level: {risk_level}")
    
    if risk_level in ["LOW", "MEDIUM"]:
        print(f"\n✅ Good news! Train and test are similar enough.")
        print(f"   Your models should generalize reasonably well.")
        print(f"\n💡 Recommendations:")
        print(f"   1. Proceed with multi-modal model")
        print(f"   2. Use all features (they're safe)")
        print(f"   3. Focus on adding images + text embeddings")
    else:
        print(f"\n⚠️  WARNING: Significant distribution shift detected!")
        print(f"   This may explain your 62% LB vs 30% CV gap.")
        print(f"\n💡 Recommendations:")
        print(f"   1. Focus on features with LOW importance in adversarial model")
        print(f"   2. Consider domain adaptation techniques")
        print(f"   3. Build robust ensemble with diverse models")
        print(f"   4. Test on holdout before submitting")
    
    # Safe features
    safe_features = feature_importance[feature_importance['importance'] < 50]['feature'].tolist()
    risky_features = feature_importance[feature_importance['importance'] > 100]['feature'].tolist()
    
    print(f"\n📋 Feature Safety:")
    print(f"   Safe features (use confidently): {len(safe_features)}")
    print(f"   Risky features (use with caution): {len(risky_features)}")
    
    if risky_features:
        print(f"\n   ⚠️  Risky features to watch:")
        for feat in risky_features[:5]:
            print(f"      - {feat}")
    
    # Save results
    print(f"\n💾 Saving results...")
    
    feature_importance.to_csv('outputs/adversarial_feature_importance.csv', index=False)
    coverage_df.to_csv('outputs/feature_coverage_comparison.csv', index=False)
    
    # Summary report
    summary = {
        'mean_auc': mean_auc,
        'std_auc': std_auc,
        'risk_level': risk_level,
        'n_safe_features': len(safe_features),
        'n_risky_features': len(risky_features)
    }
    
    pd.DataFrame([summary]).to_csv('outputs/adversarial_summary.csv', index=False)
    
    print(f"✅ Results saved:")
    print(f"   - outputs/adversarial_feature_importance.csv")
    print(f"   - outputs/feature_coverage_comparison.csv")
    print(f"   - outputs/adversarial_summary.csv")
    
    # Final verdict
    print(f"\n\n{'='*60}")
    print(f"✅ ADVERSARIAL VALIDATION COMPLETE")
    print(f"{'='*60}")
    
    print(f"\n📊 Final Verdict:")
    print(f"   AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"   Risk Level: {risk_level}")
    
    if mean_auc < 0.60:
        print(f"\n   ✅ Your features should transfer well to test set!")
    else:
        print(f"\n   ⚠️  Some features may not generalize - proceed with caution")
    
    print(f"\n💡 Next: Check feature stability and create holdout set")


if __name__ == '__main__':
    main()
