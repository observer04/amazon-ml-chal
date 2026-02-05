"""
Multi-Modal Ensemble Training - Option B
Train 3 separate branches:
1. LightGBM on tabular features (30 dims)
2. MLP on CLIP text embeddings (512 dims)
3. MLP on CLIP image embeddings (512 dims)

Then optimize ensemble weights and generate submission.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from scipy.optimize import minimize
import sys
import os

# Add config to path
sys.path.append('config')
from config import PATHS

print("=" * 80)
print("MULTI-MODAL ENSEMBLE TRAINING - OPTION B")
print("=" * 80)

# ============================================================================
# SECTION 1: LOAD ALL DATA
# ============================================================================
print("\n[1/8] Loading embeddings and features...")

# Load CLIP embeddings
train_img_emb = np.load('outputs/train_image_embeddings_clip.npy')
train_txt_emb = np.load('outputs/train_text_embeddings_clip.npy')
test_img_emb = np.load('outputs/test_image_embeddings_clip.npy')
test_txt_emb = np.load('outputs/test_text_embeddings_clip.npy')

print(f"  ✓ Image embeddings: Train {train_img_emb.shape}, Test {test_img_emb.shape}")
print(f"  ✓ Text embeddings: Train {train_txt_emb.shape}, Test {test_txt_emb.shape}")

# Load tabular features
train_df = pd.read_csv('dataset/train_with_features.csv')
test_df = pd.read_csv('dataset/test_with_features.csv')

# Select the 30 engineered features (from baseline)
feature_cols = [
    # IPQ features (most important)
    'ipq_value', 'ipq_unit_encoded', 'has_ipq',
    # Quality keywords
    'has_premium', 'has_organic', 'has_natural', 'has_pack',
    # Brand
    'has_brand',
    # Text statistics
    'text_length', 'word_count', 'avg_word_length',
    'num_bullets', 'has_bullets', 'num_lines', 'has_description',
    # Pack detection
    'has_count_pattern', 'pack_size',
    # Additional unit features
    'unit_ounce', 'unit_pound', 'unit_gram', 'unit_count',
    'unit_fluid_ounce', 'unit_liter', 'unit_piece',
    # Value tiers
    'value_tier_low', 'value_tier_mid', 'value_tier_high',
    # Composite features
    'value_per_unit', 'text_density', 'premium_score'
]

X_tab_train = train_df[feature_cols].fillna(0).values
X_tab_test = test_df[feature_cols].fillna(0).values
y_train = train_df['price'].values

print(f"  ✓ Tabular features: Train {X_tab_train.shape}, Test {X_tab_test.shape}")
print(f"  ✓ Target: {y_train.shape}, range [{y_train.min():.2f}, {y_train.max():.2f}]")

# Apply sqrt transform to target
y_train_sqrt = np.sqrt(y_train)
print(f"  ✓ Sqrt-transformed target range: [{y_train_sqrt.min():.2f}, {y_train_sqrt.max():.2f}]")

# ============================================================================
# SECTION 2: CREATE TRAIN/VAL SPLIT
# ============================================================================
print("\n[2/8] Creating train/validation split (80/20)...")

# Use same random seed as baseline for consistency
indices = np.arange(len(y_train))
train_idx, val_idx = train_test_split(
    indices, 
    test_size=0.2, 
    random_state=42
)

print(f"  ✓ Train samples: {len(train_idx)}")
print(f"  ✓ Val samples: {len(val_idx)}")

# Split all modalities
X_img_train_split, X_img_val = train_img_emb[train_idx], train_img_emb[val_idx]
X_txt_train_split, X_txt_val = train_txt_emb[train_idx], train_txt_emb[val_idx]
X_tab_train_split, X_tab_val = X_tab_train[train_idx], X_tab_train[val_idx]
y_train_split, y_val = y_train_sqrt[train_idx], y_train_sqrt[val_idx]

# ============================================================================
# SECTION 3: DEFINE MLP ARCHITECTURE
# ============================================================================
print("\n[3/8] Defining MLP architecture...")

class EmbeddingMLP(nn.Module):
    """
    MLP for CLIP embeddings (512-dim) → price prediction
    Architecture: 512 → 256 → 128 → 1
    """
    def __init__(self, input_dim=512, hidden_dims=[256, 128], dropout_rates=[0.3, 0.2]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim, dropout in zip(hidden_dims, dropout_rates):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x).squeeze(-1)

print("  ✓ MLP Architecture: [512 → 256 ReLU Drop(0.3) → 128 ReLU Drop(0.2) → 1]")

# ============================================================================
# SECTION 4: TRAIN IMAGE MLP (MOST IMPORTANT BRANCH)
# ============================================================================
print("\n[4/8] Training Image MLP (Branch 3)...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  Using device: {device}")

# Create datasets
train_dataset_img = TensorDataset(
    torch.FloatTensor(X_img_train_split),
    torch.FloatTensor(y_train_split)
)
val_dataset_img = TensorDataset(
    torch.FloatTensor(X_img_val),
    torch.FloatTensor(y_val)
)

train_loader_img = DataLoader(train_dataset_img, batch_size=256, shuffle=True)
val_loader_img = DataLoader(val_dataset_img, batch_size=512)

# Initialize model
image_mlp = EmbeddingMLP().to(device)
optimizer_img = optim.Adam(image_mlp.parameters(), lr=0.001, weight_decay=1e-5)
criterion = nn.MSELoss()

# Training loop
best_val_loss = float('inf')
patience = 10
patience_counter = 0
n_epochs = 100

print("  Training...")
for epoch in range(n_epochs):
    # Train
    image_mlp.train()
    train_loss = 0
    for X_batch, y_batch in train_loader_img:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer_img.zero_grad()
        pred = image_mlp(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        optimizer_img.step()
        
        train_loss += loss.item() * len(X_batch)
    
    train_loss /= len(train_dataset_img)
    
    # Validate
    image_mlp.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader_img:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            pred = image_mlp(X_batch)
            loss = criterion(pred, y_batch)
            val_loss += loss.item() * len(X_batch)
    
    val_loss /= len(val_dataset_img)
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(image_mlp.state_dict(), 'models/image_mlp.pth')
        patience_counter = 0
    else:
        patience_counter += 1
    
    if (epoch + 1) % 10 == 0:
        print(f"    Epoch {epoch+1}/{n_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    if patience_counter >= patience:
        print(f"  ✓ Early stopping at epoch {epoch+1}")
        break

# Load best model
image_mlp.load_state_dict(torch.load('models/image_mlp.pth'))
print(f"  ✓ Best validation MSE: {best_val_loss:.4f}")

# Calculate SMAPE on validation
image_mlp.eval()
with torch.no_grad():
    val_pred_img = image_mlp(torch.FloatTensor(X_img_val).to(device)).cpu().numpy()

# Reverse sqrt transform
val_pred_img_price = val_pred_img ** 2
y_val_price = y_val ** 2

smape_img = 100 * np.mean(2 * np.abs(val_pred_img_price - y_val_price) / (np.abs(val_pred_img_price) + np.abs(y_val_price)))
print(f"  ✓ Image MLP Validation SMAPE: {smape_img:.2f}%")

# ============================================================================
# SECTION 5: TRAIN TEXT MLP
# ============================================================================
print("\n[5/8] Training Text MLP (Branch 2)...")

# Create datasets
train_dataset_txt = TensorDataset(
    torch.FloatTensor(X_txt_train_split),
    torch.FloatTensor(y_train_split)
)
val_dataset_txt = TensorDataset(
    torch.FloatTensor(X_txt_val),
    torch.FloatTensor(y_val)
)

train_loader_txt = DataLoader(train_dataset_txt, batch_size=256, shuffle=True)
val_loader_txt = DataLoader(val_dataset_txt, batch_size=512)

# Initialize model
text_mlp = EmbeddingMLP().to(device)
optimizer_txt = optim.Adam(text_mlp.parameters(), lr=0.001, weight_decay=1e-5)

# Training loop
best_val_loss = float('inf')
patience_counter = 0

print("  Training...")
for epoch in range(n_epochs):
    # Train
    text_mlp.train()
    train_loss = 0
    for X_batch, y_batch in train_loader_txt:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer_txt.zero_grad()
        pred = text_mlp(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        optimizer_txt.step()
        
        train_loss += loss.item() * len(X_batch)
    
    train_loss /= len(train_dataset_txt)
    
    # Validate
    text_mlp.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader_txt:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            pred = text_mlp(X_batch)
            loss = criterion(pred, y_batch)
            val_loss += loss.item() * len(X_batch)
    
    val_loss /= len(val_dataset_txt)
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(text_mlp.state_dict(), 'models/text_mlp.pth')
        patience_counter = 0
    else:
        patience_counter += 1
    
    if (epoch + 1) % 10 == 0:
        print(f"    Epoch {epoch+1}/{n_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    if patience_counter >= patience:
        print(f"  ✓ Early stopping at epoch {epoch+1}")
        break

# Load best model
text_mlp.load_state_dict(torch.load('models/text_mlp.pth'))
print(f"  ✓ Best validation MSE: {best_val_loss:.4f}")

# Calculate SMAPE on validation
text_mlp.eval()
with torch.no_grad():
    val_pred_txt = text_mlp(torch.FloatTensor(X_txt_val).to(device)).cpu().numpy()

val_pred_txt_price = val_pred_txt ** 2
smape_txt = 100 * np.mean(2 * np.abs(val_pred_txt_price - y_val_price) / (np.abs(val_pred_txt_price) + np.abs(y_val_price)))
print(f"  ✓ Text MLP Validation SMAPE: {smape_txt:.2f}%")

# ============================================================================
# SECTION 6: TRAIN TABULAR LIGHTGBM
# ============================================================================
print("\n[6/8] Training Tabular LightGBM (Branch 1)...")

# Create LightGBM datasets
lgb_train = lgb.Dataset(X_tab_train_split, y_train_split)
lgb_val = lgb.Dataset(X_tab_val, y_val, reference=lgb_train)

# Parameters (from baseline)
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'min_child_samples': 20,
    'subsample': 0.8,
    'subsample_freq': 1,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'verbose': -1,
    'random_state': 42
}

print("  Training...")
lgb_model = lgb.train(
    params,
    lgb_train,
    num_boost_round=1000,
    valid_sets=[lgb_train, lgb_val],
    valid_names=['train', 'val'],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
)

print(f"  ✓ Best iteration: {lgb_model.best_iteration}")

# Save model
lgb_model.save_model('models/tabular_lgbm.txt')

# Predict on validation
val_pred_tab = lgb_model.predict(X_tab_val, num_iteration=lgb_model.best_iteration)
val_pred_tab_price = val_pred_tab ** 2
smape_tab = 100 * np.mean(2 * np.abs(val_pred_tab_price - y_val_price) / (np.abs(val_pred_tab_price) + np.abs(y_val_price)))
print(f"  ✓ Tabular LightGBM Validation SMAPE: {smape_tab:.2f}%")

# ============================================================================
# SECTION 7: OPTIMIZE ENSEMBLE WEIGHTS
# ============================================================================
print("\n[7/8] Optimizing ensemble weights...")
print(f"\n  Individual model SMAPE scores:")
print(f"    - Tabular LGBM: {smape_tab:.2f}%")
print(f"    - Text MLP:     {smape_txt:.2f}%")
print(f"    - Image MLP:    {smape_img:.2f}%")

def ensemble_smape(weights, pred_tab, pred_txt, pred_img, y_true):
    """Calculate SMAPE for weighted ensemble"""
    w_tab, w_txt, w_img = weights
    ensemble_pred = w_tab * pred_tab + w_txt * pred_txt + w_img * pred_img
    smape = 100 * np.mean(2 * np.abs(ensemble_pred - y_true) / (np.abs(ensemble_pred) + np.abs(y_true)))
    return smape

# Optimize weights
def objective(weights):
    return ensemble_smape(weights, val_pred_tab_price, val_pred_txt_price, val_pred_img_price, y_val_price)

# Constraints: weights sum to 1, all non-negative
constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
bounds = [(0, 1), (0, 1), (0, 1)]
initial_weights = [0.2, 0.3, 0.5]  # Start with educated guess

print(f"\n  Initial weights [tab, txt, img]: {initial_weights}")
print(f"  Initial ensemble SMAPE: {objective(initial_weights):.2f}%")

result = minimize(
    objective,
    initial_weights,
    method='SLSQP',
    bounds=bounds,
    constraints=constraints
)

optimal_weights = result.x
print(f"\n  ✓ Optimized weights [tab, txt, img]: [{optimal_weights[0]:.3f}, {optimal_weights[1]:.3f}, {optimal_weights[2]:.3f}]")
print(f"  ✓ Ensemble Validation SMAPE: {result.fun:.2f}%")

# Calculate improvement
best_single_smape = min(smape_tab, smape_txt, smape_img)
improvement = best_single_smape - result.fun
print(f"\n  📊 Improvement over best single model: {improvement:.2f}% SMAPE")

# ============================================================================
# SECTION 8: GENERATE TEST PREDICTIONS
# ============================================================================
print("\n[8/8] Generating test predictions...")

# Predict on full train for final models
print("  Re-training on full training set...")

# Image MLP on full train
train_dataset_img_full = TensorDataset(
    torch.FloatTensor(train_img_emb),
    torch.FloatTensor(y_train_sqrt)
)
train_loader_img_full = DataLoader(train_dataset_img_full, batch_size=256, shuffle=True)

image_mlp_final = EmbeddingMLP().to(device)
optimizer_img_final = optim.Adam(image_mlp_final.parameters(), lr=0.001, weight_decay=1e-5)

for epoch in range(50):  # Fewer epochs, no validation
    image_mlp_final.train()
    for X_batch, y_batch in train_loader_img_full:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer_img_final.zero_grad()
        pred = image_mlp_final(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        optimizer_img_final.step()

# Text MLP on full train
train_dataset_txt_full = TensorDataset(
    torch.FloatTensor(train_txt_emb),
    torch.FloatTensor(y_train_sqrt)
)
train_loader_txt_full = DataLoader(train_dataset_txt_full, batch_size=256, shuffle=True)

text_mlp_final = EmbeddingMLP().to(device)
optimizer_txt_final = optim.Adam(text_mlp_final.parameters(), lr=0.001, weight_decay=1e-5)

for epoch in range(50):
    text_mlp_final.train()
    for X_batch, y_batch in train_loader_txt_full:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer_txt_final.zero_grad()
        pred = text_mlp_final(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        optimizer_txt_final.step()

# LightGBM on full train
lgb_train_full = lgb.Dataset(X_tab_train, y_train_sqrt)
lgb_model_final = lgb.train(
    params,
    lgb_train_full,
    num_boost_round=lgb_model.best_iteration
)

print("  ✓ Final models trained on full dataset")

# Predict on test set
print("  Generating test predictions...")

image_mlp_final.eval()
text_mlp_final.eval()

with torch.no_grad():
    test_pred_img = image_mlp_final(torch.FloatTensor(test_img_emb).to(device)).cpu().numpy()
    test_pred_txt = text_mlp_final(torch.FloatTensor(test_txt_emb).to(device)).cpu().numpy()

test_pred_tab = lgb_model_final.predict(X_tab_test)

# Reverse sqrt transform
test_pred_img_price = test_pred_img ** 2
test_pred_txt_price = test_pred_txt ** 2
test_pred_tab_price = test_pred_tab ** 2

# Apply optimal ensemble weights
test_pred_ensemble = (
    optimal_weights[0] * test_pred_tab_price +
    optimal_weights[1] * test_pred_txt_price +
    optimal_weights[2] * test_pred_img_price
)

# Clip negative predictions
test_pred_ensemble = np.maximum(test_pred_ensemble, 0.01)

print(f"  ✓ Test predictions range: [{test_pred_ensemble.min():.2f}, {test_pred_ensemble.max():.2f}]")

# ============================================================================
# SAVE SUBMISSION
# ============================================================================
submission = pd.DataFrame({
    'sample_id': test_df['sample_id'],
    'price': test_pred_ensemble
})

submission.to_csv('submission_multimodal_ensemble.csv', index=False)
print(f"\n✅ Submission saved: submission_multimodal_ensemble.csv")
print(f"   Shape: {submission.shape}")
print(f"   Sample predictions:")
print(submission.head(10))

print("\n" + "=" * 80)
print("MULTI-MODAL ENSEMBLE TRAINING COMPLETE!")
print("=" * 80)
print(f"\n📊 FINAL RESULTS:")
print(f"   Validation SMAPE: {result.fun:.2f}%")
print(f"   Ensemble weights: Tab={optimal_weights[0]:.3f}, Txt={optimal_weights[1]:.3f}, Img={optimal_weights[2]:.3f}")
print(f"   Next step: Validate and submit to Kaggle!")
print("=" * 80)
