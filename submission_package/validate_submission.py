"""
Quick Validation Script - Run Before Submission
===============================================

This script does final sanity checks on the submission file.
Run this locally before uploading to Kaggle.

Author: observer04
Date: October 13, 2025
"""

import pandas as pd
import numpy as np

print("="*80)
print("SUBMISSION VALIDATION CHECKS")
print("="*80)

# Load submission
submission = pd.read_csv('submission_enhanced.csv')

print("\n[1/5] Format Check")
print(f"  Rows: {len(submission):,}")
print(f"  Columns: {list(submission.columns)}")

assert len(submission) == 75000, "ERROR: Should have 75,000 rows!"
assert list(submission.columns) == ['sample_id', 'price'], "ERROR: Wrong column names!"
print("  ✅ Format correct")

print("\n[2/5] Sample ID Check")
test_df = pd.read_csv('dataset/test_with_features.csv')
expected_ids = set(test_df['sample_id'])
submission_ids = set(submission['sample_id'])

assert expected_ids == submission_ids, "ERROR: Sample IDs don't match test set!"
print(f"  ✅ All {len(expected_ids):,} sample IDs match")

print("\n[3/5] Price Range Check")
prices = submission['price'].values
print(f"  Min:    ${prices.min():.2f}")
print(f"  Max:    ${prices.max():.2f}")
print(f"  Mean:   ${prices.mean():.2f}")
print(f"  Median: ${np.median(prices):.2f}")
print(f"  Std:    ${prices.std():.2f}")

# Check for invalid values
assert not submission['price'].isna().any(), "ERROR: NaN prices found!"
assert (prices >= 0).all(), "ERROR: Negative prices found!"
assert (prices <= 10000).all(), "WARNING: Extremely high prices (>$10k)"
print("  ✅ All prices valid")

print("\n[4/5] Distribution Check")
train_df = pd.read_csv('dataset/train_with_features.csv', low_memory=False)
train_prices = train_df['price'].values

print(f"  Train mean:   ${train_prices.mean():.2f}")
print(f"  Submit mean:  ${prices.mean():.2f}")
print(f"  Difference:   ${prices.mean() - train_prices.mean():.2f}")

print(f"\n  Train median: ${np.median(train_prices):.2f}")
print(f"  Submit median: ${np.median(prices):.2f}")
print(f"  Difference:   ${np.median(prices) - np.median(train_prices):.2f}")

# Distribution bins
bins = [0, 10, 25, 50, 100, np.inf]
labels = ['<$10', '$10-25', '$25-50', '$50-100', '>$100']

train_dist = pd.cut(train_prices, bins=bins, labels=labels).value_counts(normalize=True) * 100
submit_dist = pd.cut(prices, bins=bins, labels=labels).value_counts(normalize=True) * 100

print(f"\n  Distribution Comparison:")
for label in labels:
    train_pct = train_dist.get(label, 0)
    submit_pct = submit_dist.get(label, 0)
    diff = submit_pct - train_pct
    status = "✅" if abs(diff) < 10 else "⚠️"
    print(f"    {label:10s}: Train {train_pct:5.1f}% | Submit {submit_pct:5.1f}% | Diff {diff:+5.1f}% {status}")

print("\n[5/5] Outlier Check")
extreme_low = (prices < 0.1).sum()
extreme_high = (prices > 500).sum()
print(f"  Prices < $0.10: {extreme_low} ({extreme_low/len(prices)*100:.2f}%)")
print(f"  Prices > $500: {extreme_high} ({extreme_high/len(prices)*100:.2f}%)")

if extreme_low > 100:
    print("  ⚠️  Many very low prices - check if reasonable")
if extreme_high > 100:
    print("  ⚠️  Many very high prices - check if reasonable")

print("\n" + "="*80)
print("VALIDATION COMPLETE")
print("="*80)
print("\n✅ Ready to submit to Kaggle!")
print("\nExpected Leaderboard Score: 28-31% SMAPE")
print("(Allowing 1-2% gap between validation and leaderboard)")
