"""
Baseline Check - Simple Heuristic Predictor
============================================

Purpose: Establish minimum performance bar using simple rule-based prediction.

This script creates a baseline model that predicts price based on:
1. Median price per unit type (e.g., Ounce, Count, Fl Oz)
2. Global median fallback for missing units

Why this matters:
- Establishes minimum viable performance
- Any ML model MUST beat this baseline
- If complex model < baseline SMAPE, something is wrong!

Expected performance: 50-55% SMAPE
"""

import pandas as pd
import numpy as np
from pathlib import Path

def calculate_smape(y_true, y_pred):
    """Calculate Symmetric Mean Absolute Percentage Error."""
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    # Avoid division by zero
    denominator = np.where(denominator == 0, 1e-10, denominator)
    return 100 * np.mean(np.abs(y_pred - y_true) / denominator)


class BaselinePredictor:
    """Simple baseline: Predict median price per unit type."""
    
    def __init__(self):
        self.unit_medians = {}
        self.global_median = None
        
    def fit(self, df):
        """Learn median price per unit from training data."""
        # Calculate median price per unit
        self.unit_medians = df.groupby('unit')['price'].median().to_dict()
        
        # Global median fallback
        self.global_median = df['price'].median()
        
        print(f"\n📊 Learned {len(self.unit_medians)} unit types:")
        for unit, median_price in sorted(self.unit_medians.items(), 
                                        key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {unit:15s}: ${median_price:7.2f}")
        
        print(f"\n🌍 Global median (fallback): ${self.global_median:.2f}")
        
    def predict(self, df):
        """Predict prices using unit medians."""
        predictions = df['unit'].map(self.unit_medians)
        
        # Fill missing units with global median
        predictions = predictions.fillna(self.global_median)
        
        return predictions.values
    
    def evaluate(self, df):
        """Evaluate baseline performance on validation set."""
        predictions = self.predict(df)
        smape = calculate_smape(df['price'].values, predictions)
        
        # Calculate by price segment
        df_eval = df.copy()
        df_eval['prediction'] = predictions
        
        print(f"\n{'='*60}")
        print(f"BASELINE PERFORMANCE")
        print(f"{'='*60}")
        print(f"\nOverall SMAPE: {smape:.2f}%")
        
        # Breakdown by price range
        print(f"\n📊 Performance by Price Range:")
        price_ranges = [
            ('Budget (<$10)', df_eval['price'] < 10),
            ('Mid-Range ($10-$50)', (df_eval['price'] >= 10) & (df_eval['price'] < 50)),
            ('Premium ($50-$100)', (df_eval['price'] >= 50) & (df_eval['price'] < 100)),
            ('Luxury (>$100)', df_eval['price'] >= 100)
        ]
        
        for name, mask in price_ranges:
            if mask.sum() > 0:
                subset_smape = calculate_smape(
                    df_eval[mask]['price'].values,
                    df_eval[mask]['prediction'].values
                )
                count = mask.sum()
                pct = count / len(df_eval) * 100
                print(f"  {name:25s}: {subset_smape:6.2f}% SMAPE (n={count:,}, {pct:.1f}%)")
        
        # Prediction statistics
        print(f"\n📈 Prediction Statistics:")
        print(f"  Mean prediction: ${predictions.mean():.2f}")
        print(f"  Median prediction: ${np.median(predictions):.2f}")
        print(f"  Min prediction: ${predictions.min():.2f}")
        print(f"  Max prediction: ${predictions.max():.2f}")
        
        print(f"\n✅ Baseline SMAPE: {smape:.2f}%")
        print(f"🎯 Your ML model must achieve <{smape:.0f}% to be useful!")
        
        return smape


def main():
    """Run baseline check."""
    print("="*60)
    print("BASELINE CHECK: Simple Heuristic Predictor")
    print("="*60)
    
    # Load training data with features
    print("\n📂 Loading training data...")
    train_df = pd.read_csv('dataset/train_with_features.csv')
    print(f"✅ Loaded {len(train_df):,} training samples")
    
    # Filter to samples with valid unit (84.8%)
    train_valid = train_df[train_df['unit'].notna()].copy()
    print(f"✅ Using {len(train_valid):,} samples with valid units ({len(train_valid)/len(train_df)*100:.1f}%)")
    
    # Split train into train/val (80/20)
    from sklearn.model_selection import train_test_split
    
    train_split, val_split = train_test_split(
        train_valid, 
        test_size=0.2, 
        random_state=42,
        stratify=pd.cut(train_valid['price'], bins=[0, 10, 50, 100, 3000])
    )
    
    print(f"\n📊 Split:")
    print(f"  Train: {len(train_split):,} samples")
    print(f"  Val:   {len(val_split):,} samples")
    
    # Train baseline
    print(f"\n🔧 Training baseline predictor...")
    baseline = BaselinePredictor()
    baseline.fit(train_split)
    
    # Evaluate on validation
    print(f"\n🧪 Evaluating on validation set...")
    val_smape = baseline.evaluate(val_split)
    
    # Cross-validation estimate
    print(f"\n\n{'='*60}")
    print(f"5-FOLD CROSS-VALIDATION")
    print(f"{'='*60}")
    
    from sklearn.model_selection import KFold
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_smapes = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_valid), 1):
        fold_train = train_valid.iloc[train_idx]
        fold_val = train_valid.iloc[val_idx]
        
        fold_baseline = BaselinePredictor()
        fold_baseline.fit(fold_train)
        
        fold_preds = fold_baseline.predict(fold_val)
        fold_smape = calculate_smape(fold_val['price'].values, fold_preds)
        cv_smapes.append(fold_smape)
        
        print(f"Fold {fold}: {fold_smape:.2f}% SMAPE")
    
    print(f"\n{'='*60}")
    print(f"Mean CV SMAPE: {np.mean(cv_smapes):.2f}% ± {np.std(cv_smapes):.2f}%")
    print(f"{'='*60}")
    
    # Save baseline model
    import pickle
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / 'baseline_model.pkl', 'wb') as f:
        pickle.dump(baseline, f)
    
    print(f"\n💾 Baseline model saved to outputs/baseline_model.pkl")
    
    # Generate test predictions
    print(f"\n🔮 Generating test predictions...")
    test_df = pd.read_csv('dataset/test_with_features.csv')
    test_preds = baseline.predict(test_df)
    
    # Create submission file
    submission = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': test_preds
    })
    
    submission_path = output_dir / 'baseline_submission.csv'
    submission.to_csv(submission_path, index=False)
    
    print(f"✅ Test predictions saved to {submission_path}")
    print(f"\nTest prediction statistics:")
    print(f"  Mean: ${test_preds.mean():.2f}")
    print(f"  Median: ${np.median(test_preds):.2f}")
    print(f"  Range: ${test_preds.min():.2f} - ${test_preds.max():.2f}")
    
    # Final summary
    print(f"\n\n{'='*60}")
    print(f"✅ BASELINE CHECK COMPLETE")
    print(f"{'='*60}")
    print(f"\n📊 Key Metrics:")
    print(f"  Validation SMAPE: {val_smape:.2f}%")
    print(f"  CV SMAPE: {np.mean(cv_smapes):.2f}% ± {np.std(cv_smapes):.2f}%")
    print(f"\n🎯 Minimum Bar to Beat: {np.mean(cv_smapes):.0f}% SMAPE")
    print(f"\n💡 Next Steps:")
    print(f"  1. Run adversarial validation (train vs test)")
    print(f"  2. Check feature stability")
    print(f"  3. Build multi-modal model")
    print(f"  4. Ensure ML model beats {np.mean(cv_smapes):.0f}% SMAPE!")
    

if __name__ == '__main__':
    main()
