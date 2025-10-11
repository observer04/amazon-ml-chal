"""
FINAL VALIDATION SCRIPT
Validates test_out.csv against all submission requirements
"""

import pandas as pd
import numpy as np

print("\n" + "="*70)
print("SUBMISSION VALIDATION - Amazon ML Challenge")
print("="*70 + "\n")

# Load files
print("Loading files...")
try:
    df_submission = pd.read_csv('test_out.csv')
    df_test = pd.read_csv('dataset/test.csv')
    df_sample = pd.read_csv('dataset/sample_test_out.csv')
    print("✅ All files loaded\n")
except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    exit(1)

# ============================================================================
# VALIDATION CHECKS
# ============================================================================

checks_passed = []
checks_failed = []

print("="*70)
print("FORMAT VALIDATION")
print("="*70)

# Check 1: Columns
expected_cols = ['sample_id', 'price']
if list(df_submission.columns) == expected_cols:
    checks_passed.append("✅ Column names: ['sample_id', 'price']")
else:
    checks_failed.append(f"❌ Column names: {list(df_submission.columns)} (expected {expected_cols})")

# Check 2: Number of rows
if len(df_submission) == 75000:
    checks_passed.append(f"✅ Row count: 75,000")
else:
    checks_failed.append(f"❌ Row count: {len(df_submission):,} (expected 75,000)")

# Check 3: No missing values
missing = df_submission.isnull().sum().sum()
if missing == 0:
    checks_passed.append("✅ No missing values")
else:
    checks_failed.append(f"❌ Missing values: {missing}")

# Check 4: All prices positive
if (df_submission['price'] > 0).all():
    checks_passed.append("✅ All prices positive")
else:
    neg_count = (df_submission['price'] <= 0).sum()
    checks_failed.append(f"❌ Non-positive prices: {neg_count}")

# Check 5: Prices are floats
if df_submission['price'].dtype in ['float64', 'float32']:
    checks_passed.append("✅ Prices are float type")
else:
    checks_failed.append(f"❌ Price dtype: {df_submission['price'].dtype} (expected float)")

# Check 6: Sample IDs match test.csv
if set(df_submission['sample_id']) == set(df_test['sample_id']):
    checks_passed.append("✅ Sample IDs match test.csv")
else:
    missing_ids = set(df_test['sample_id']) - set(df_submission['sample_id'])
    extra_ids = set(df_submission['sample_id']) - set(df_test['sample_id'])
    if missing_ids:
        checks_failed.append(f"❌ Missing {len(missing_ids)} sample IDs")
    if extra_ids:
        checks_failed.append(f"❌ Extra {len(extra_ids)} sample IDs not in test.csv")

# Check 7: No duplicate sample IDs
if df_submission['sample_id'].nunique() == len(df_submission):
    checks_passed.append("✅ No duplicate sample IDs")
else:
    dupes = len(df_submission) - df_submission['sample_id'].nunique()
    checks_failed.append(f"❌ Duplicate sample IDs: {dupes}")

# Check 8: Format matches sample_test_out.csv
if df_submission.dtypes.to_dict() == df_sample.dtypes.to_dict():
    checks_passed.append("✅ Data types match sample_test_out.csv")
else:
    checks_failed.append("⚠️  Data types slightly different (but likely OK)")

for check in checks_passed:
    print(check)
for check in checks_failed:
    print(check)

print("\n" + "="*70)
print("PREDICTION STATISTICS")
print("="*70)

print(f"Min price:        ${df_submission['price'].min():.2f}")
print(f"Max price:        ${df_submission['price'].max():.2f}")
print(f"Mean price:       ${df_submission['price'].mean():.2f}")
print(f"Median price:     ${df_submission['price'].median():.2f}")
print(f"Std dev:          ${df_submission['price'].std():.2f}")
print(f"\nPrice distribution:")
print(f"  < $10:          {(df_submission['price'] < 10).sum():,} ({(df_submission['price'] < 10).sum()/len(df_submission)*100:.1f}%)")
print(f"  $10 - $50:      {((df_submission['price'] >= 10) & (df_submission['price'] < 50)).sum():,} ({((df_submission['price'] >= 10) & (df_submission['price'] < 50)).sum()/len(df_submission)*100:.1f}%)")
print(f"  $50 - $100:     {((df_submission['price'] >= 50) & (df_submission['price'] < 100)).sum():,} ({((df_submission['price'] >= 50) & (df_submission['price'] < 100)).sum()/len(df_submission)*100:.1f}%)")
print(f"  > $100:         {(df_submission['price'] >= 100).sum():,} ({(df_submission['price'] >= 100).sum()/len(df_submission)*100:.1f}%)")

print("\n" + "="*70)
print("SANITY CHECKS")
print("="*70)

sanity_passed = []
sanity_warnings = []

# Sanity 1: Reasonable price range
if df_submission['price'].max() < 5000:
    sanity_passed.append("✅ Max price reasonable (<$5000)")
else:
    sanity_warnings.append(f"⚠️  Max price high: ${df_submission['price'].max():.2f}")

# Sanity 2: Min price not too low
if df_submission['price'].min() >= 0.5:
    sanity_passed.append("✅ Min price reasonable (≥$0.50)")
else:
    sanity_warnings.append(f"⚠️  Min price very low: ${df_submission['price'].min():.2f}")

# Sanity 3: Mean price reasonable
if 10 <= df_submission['price'].mean() <= 50:
    sanity_passed.append("✅ Mean price reasonable ($10-$50)")
else:
    sanity_warnings.append(f"⚠️  Mean price: ${df_submission['price'].mean():.2f}")

# Sanity 4: Check for constant predictions
unique_ratio = df_submission['price'].nunique() / len(df_submission)
if unique_ratio > 0.8:
    sanity_passed.append(f"✅ Diverse predictions ({df_submission['price'].nunique():,} unique)")
else:
    sanity_warnings.append(f"⚠️  Low diversity: {df_submission['price'].nunique():,} unique values")

# Sanity 5: No extreme outliers
q99 = df_submission['price'].quantile(0.99)
if q99 < 500:
    sanity_passed.append(f"✅ 99th percentile reasonable: ${q99:.2f}")
else:
    sanity_warnings.append(f"⚠️  99th percentile high: ${q99:.2f}")

for check in sanity_passed:
    print(check)
for check in sanity_warnings:
    print(check)

print("\n" + "="*70)
print("FINAL VERDICT")
print("="*70)

if len(checks_failed) == 0:
    print("🎉 SUBMISSION IS VALID AND READY!")
    print("✅ All format checks passed")
    print("✅ All sanity checks passed or have minor warnings")
    print("\n📤 You can submit test_out.csv to the portal")
    
    print("\n" + "="*70)
    print("SUBMISSION CHECKLIST")
    print("="*70)
    print("□ Upload test_out.csv to portal")
    print("□ Prepare 1-page documentation (use Documentation_template.md)")
    print("  - Model: LightGBM gradient boosting")
    print("  - Features: 28 features (IPQ + quality + brand + interactions)")
    print("  - Transform: Square root transform")
    print("  - Validation: 5-fold CV, ~30% SMAPE")
    print("□ Note Public LB score after submission")
    print("□ Compare LB score vs CV score (30.70%)")
else:
    print("❌ SUBMISSION HAS ERRORS!")
    print(f"   {len(checks_failed)} critical issues found")
    print("   Fix these before submitting:")
    for check in checks_failed:
        print(f"   {check}")

print("\n" + "="*70 + "\n")
