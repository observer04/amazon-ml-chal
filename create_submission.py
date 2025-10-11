"""
SUBMISSION SCRIPT - Amazon ML Challenge
Trains best model (Exp 4) and generates test predictions
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_PATH = 'dataset'
RANDOM_STATE = 42


def train_final_model(df_train, features):
    """
    Train final model on ALL training data using 5-fold CV
    Returns: List of 5 trained models (one per fold)
    """
    print("="*70)
    print("TRAINING FINAL MODEL - 5-FOLD CV")
    print("="*70)
    print(f"Features: {len(features)}")
    print(f"Training samples: {len(df_train):,}\n")
    
    # Encode categorical features
    label_encoders = {}
    for col in features:
        if df_train[col].dtype == 'object':
            le = LabelEncoder()
            df_train[col] = df_train[col].fillna('missing')
            df_train[col] = le.fit_transform(df_train[col].astype(str))
            label_encoders[col] = le
    
    X = df_train[features].values
    y = np.sqrt(df_train['price'].values)  # SQRT transform
    
    # 5-Fold CV to train 5 models
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    models = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        print(f"Training Fold {fold}/5...", end=" ")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # LightGBM params - AGGRESSIVE TRAINING FOR FINAL MODEL
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'num_leaves': 127,             # Increased from 63 (more complex)
            'learning_rate': 0.02,         # Slower (was 0.03)
            'min_data_in_leaf': 10,        # Finer splits (was 15)
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'lambda_l1': 0.3,              # Lighter regularization
            'lambda_l2': 0.3,
            'max_depth': 12,               # Limit depth to prevent overfitting
            'verbose': -1,
            'seed': RANDOM_STATE,
        }
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=3000,          # Increased from 2000
            valid_sets=[val_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100),  # More patience (was 50)
                lgb.log_evaluation(period=0)
            ]
        )
        
        models.append(model)
        print(f"✓ Best iteration: {model.best_iteration}")
    
    print(f"\n✅ Trained {len(models)} models\n")
    return models, label_encoders


def predict_test(df_test, features, models, label_encoders):
    """
    Generate predictions on test set using ensemble of 5 models
    """
    print("="*70)
    print("GENERATING TEST PREDICTIONS")
    print("="*70)
    print(f"Test samples: {len(df_test):,}\n")
    
    # Encode categorical features (same encoding as train)
    for col in features:
        if col in label_encoders:
            df_test[col] = df_test[col].fillna('missing')
            # Handle unseen categories
            df_test[col] = df_test[col].apply(
                lambda x: x if x in label_encoders[col].classes_ else 'missing'
            )
            df_test[col] = label_encoders[col].transform(df_test[col].astype(str))
    
    X_test = df_test[features].values
    
    # Predict with each model and average (ensemble)
    predictions = []
    for i, model in enumerate(models, 1):
        print(f"Predicting with model {i}/5...", end=" ")
        pred_sqrt = model.predict(X_test, num_iteration=model.best_iteration)
        pred = np.maximum(pred_sqrt ** 2, 0)  # Inverse sqrt, clip negatives
        predictions.append(pred)
        print("✓")
    
    # Average predictions from 5 models
    final_pred = np.mean(predictions, axis=0)
    
    print(f"\n✅ Generated {len(final_pred):,} predictions")
    print(f"   Min: ${final_pred.min():.2f}")
    print(f"   Max: ${final_pred.max():.2f}")
    print(f"   Mean: ${final_pred.mean():.2f}")
    print(f"   Median: ${np.median(final_pred):.2f}\n")
    
    return final_pred


def validate_submission(df_submission):
    """
    Validate submission format
    """
    print("="*70)
    print("VALIDATING SUBMISSION")
    print("="*70)
    
    checks = []
    
    # Check 1: Correct columns
    if list(df_submission.columns) == ['sample_id', 'price']:
        checks.append("✅ Columns correct: ['sample_id', 'price']")
    else:
        checks.append(f"❌ Columns wrong: {list(df_submission.columns)}")
    
    # Check 2: Correct number of rows
    if len(df_submission) == 75000:
        checks.append(f"✅ Rows correct: 75,000")
    else:
        checks.append(f"❌ Rows wrong: {len(df_submission):,}")
    
    # Check 3: No missing values
    if df_submission.isnull().sum().sum() == 0:
        checks.append("✅ No missing values")
    else:
        checks.append(f"❌ Missing values: {df_submission.isnull().sum().sum()}")
    
    # Check 4: All prices positive
    if (df_submission['price'] > 0).all():
        checks.append("✅ All prices positive")
    else:
        neg_count = (df_submission['price'] <= 0).sum()
        checks.append(f"❌ Non-positive prices: {neg_count}")
    
    # Check 5: Sample IDs match test.csv
    df_test_orig = pd.read_csv(f'{DATA_PATH}/test.csv')
    if set(df_submission['sample_id']) == set(df_test_orig['sample_id']):
        checks.append("✅ Sample IDs match test.csv")
    else:
        checks.append("❌ Sample IDs don't match test.csv")
    
    for check in checks:
        print(check)
    
    all_pass = all('✅' in check for check in checks)
    
    if all_pass:
        print("\n🎉 SUBMISSION READY!")
    else:
        print("\n⚠️ SUBMISSION HAS ERRORS - FIX BEFORE SUBMITTING")
    
    print("="*70 + "\n")
    return all_pass


def main():
    print("\n" + "="*70)
    print("AMAZON ML CHALLENGE - SUBMISSION GENERATION")
    print("Model: LightGBM with SQRT transform + Interactions")
    print("Expected CV SMAPE: ~30.70%")
    print("="*70 + "\n")
    
    # Load data
    print("Loading data...")
    df_train = pd.read_csv(f'{DATA_PATH}/train_with_features.csv', low_memory=False)
    df_test = pd.read_csv(f'{DATA_PATH}/test_with_features.csv', low_memory=False)
    print(f"✅ Train: {len(df_train):,} rows")
    print(f"✅ Test: {len(df_test):,} rows\n")
    
    # Define features (same as Experiment 4)
    features = [
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
    
    # Train final models
    models, label_encoders = train_final_model(df_train, features)
    
    # Generate predictions
    predictions = predict_test(df_test, features, models, label_encoders)
    
    # Create submission file
    submission = pd.DataFrame({
        'sample_id': df_test['sample_id'],
        'price': predictions
    })
    
    # Validate
    is_valid = validate_submission(submission)
    
    if is_valid:
        # Save submission
        output_file = 'test_out.csv'
        submission.to_csv(output_file, index=False)
        print(f"✅ Submission saved to: {output_file}")
        print(f"   File size: {submission.memory_usage(deep=True).sum() / 1024:.1f} KB")
        print(f"\n📤 Ready to upload to Kaggle!")
    else:
        print("❌ Fix errors before saving submission")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    main()
