"""
Generate feature files for training and testing.

This script:
1. Loads train.csv and test.csv with proper CSV handling (handles multiline text)
2. Applies feature extraction from src/feature_extraction.py
3. Saves to train_with_features.csv and test_with_features.csv with proper escaping

IMPORTANT: Uses quoting=csv.QUOTE_ALL to handle newlines in catalog_content
"""

import pandas as pd
import sys
import time
sys.path.append('.')
from src.feature_extraction import create_features

# Data paths
DATA_PATH = 'dataset'

def load_data_safely(filepath):
    """Load CSV with proper handling of multiline text fields"""
    print(f"Loading {filepath}...")
    # Use quoting=1 (QUOTE_ALL) to handle embedded newlines
    df = pd.read_csv(filepath, quoting=1)
    print(f"✅ Loaded {len(df):,} rows, {len(df.columns)} columns")
    return df

def save_features_safely(df, filepath):
    """Save CSV with proper escaping of multiline text"""
    print(f"\nSaving to {filepath}...")
    # Use quoting=1 (QUOTE_ALL) to properly escape multiline text
    df.to_csv(filepath, index=False, quoting=1)
    print(f"✅ Saved {len(df):,} rows, {len(df.columns)} columns")
    
    # Verify file
    df_verify = pd.read_csv(filepath, quoting=1)
    if len(df_verify) != len(df):
        print(f"❌ ERROR: File has {len(df_verify)} rows, expected {len(df)}")
    else:
        print(f"✅ Verified: File has correct {len(df_verify):,} rows")

def main():
    print("="*80)
    print("FEATURE GENERATION - Train & Test Datasets")
    print("="*80)
    
    # ========================================
    # TRAIN FEATURES
    # ========================================
    print("\n" + "="*80)
    print("1️⃣  PROCESSING TRAIN.CSV")
    print("="*80)
    
    start_time = time.time()
    
    # Load train data
    train_df = load_data_safely(f'{DATA_PATH}/train.csv')
    print(f"\nColumns: {list(train_df.columns)}")
    print(f"Sample IDs: {train_df['sample_id'].min()} to {train_df['sample_id'].max()}")
    print(f"Price range: ${train_df['price'].min():.2f} to ${train_df['price'].max():.2f}")
    
    # Extract features
    print("\n" + "-"*60)
    print("Extracting features...")
    print("-"*60)
    train_features = create_features(train_df)
    
    elapsed = time.time() - start_time
    print(f"\n✅ Feature extraction complete in {elapsed:.1f} seconds")
    print(f"Features created: {len(train_features.columns)} total columns")
    
    # Show feature coverage
    print("\n" + "-"*60)
    print("Feature Coverage:")
    print("-"*60)
    feature_cols = [col for col in train_features.columns 
                    if col not in ['sample_id', 'catalog_content', 'image_link', 'price']]
    
    for col in feature_cols[:15]:  # Show first 15 features
        if train_features[col].dtype in ['int64', 'float64']:
            coverage = train_features[col].notna().sum() / len(train_features) * 100
            print(f"  {col:20s}: {coverage:5.1f}% coverage")
    
    # Save
    save_features_safely(train_features, f'{DATA_PATH}/train_with_features.csv')
    
    # ========================================
    # TEST FEATURES
    # ========================================
    print("\n" + "="*80)
    print("2️⃣  PROCESSING TEST.CSV")
    print("="*80)
    
    start_time = time.time()
    
    # Load test data
    test_df = load_data_safely(f'{DATA_PATH}/test.csv')
    print(f"\nColumns: {list(test_df.columns)}")
    print(f"Sample IDs: {test_df['sample_id'].min()} to {test_df['sample_id'].max()}")
    
    # Extract features
    print("\n" + "-"*60)
    print("Extracting features...")
    print("-"*60)
    test_features = create_features(test_df)
    
    elapsed = time.time() - start_time
    print(f"\n✅ Feature extraction complete in {elapsed:.1f} seconds")
    print(f"Features created: {len(test_features.columns)} total columns")
    
    # Show feature coverage
    print("\n" + "-"*60)
    print("Feature Coverage:")
    print("-"*60)
    feature_cols_test = [col for col in test_features.columns 
                         if col not in ['sample_id', 'catalog_content', 'image_link']]
    
    for col in feature_cols_test[:15]:  # Show first 15 features
        if test_features[col].dtype in ['int64', 'float64']:
            coverage = test_features[col].notna().sum() / len(test_features) * 100
            print(f"  {col:20s}: {coverage:5.1f}% coverage")
    
    # Save
    save_features_safely(test_features, f'{DATA_PATH}/test_with_features.csv')
    
    # ========================================
    # FINAL SUMMARY
    # ========================================
    print("\n" + "="*80)
    print("✅ FEATURE GENERATION COMPLETE")
    print("="*80)
    print(f"\nGenerated files:")
    print(f"  📄 train_with_features.csv: {len(train_features):,} rows × {len(train_features.columns)} columns")
    print(f"  📄 test_with_features.csv:  {len(test_features):,} rows × {len(test_features.columns)} columns")
    print(f"\nReady for baseline model training! 🚀")
    print(f"\nNext step: python baseline_model.py")

if __name__ == '__main__':
    main()
