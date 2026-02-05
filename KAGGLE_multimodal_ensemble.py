"""
COMPLETE MULTI-MODAL ENSEMBLE TRAINING - OPTION B
===================================================
Run this entire file in one Kaggle notebook cell.

This script trains 3 separate branches:
1. LightGBM on 30 tabular features
2. MLP on CLIP text embeddings (512-dim)
3. MLP on CLIP image embeddings (512-dim)

Then optimizes ensemble weights and generates submission.

GPU Required: Yes (for MLP training)
Time Estimate: 30-60 minutes
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
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("MULTI-MODAL ENSEMBLE TRAINING - OPTION B")
print("Training 3 branches: Tabular LGBM + Text MLP + Image MLP")
print("=" * 80)

# ============================================================================
# SECTION 1: LOAD ALL DATA
# ============================================================================
print("\n[STEP 1/8] Loading embeddings and features...")

# Load CLIP embeddings (generated in previous step)
train_img_emb = np.load('/kaggle/working/outputs/train_image_embeddings_clip.npy')
train_txt_emb = np.load('/kaggle/working/outputs/train_text_embeddings_clip.npy')
test_img_emb = np.load('/kaggle/working/outputs/test_image_embeddings_clip.npy')
test_txt_emb = np.load('/kaggle/working/outputs/test_text_embeddings_clip.npy')

print(f"✓ Image embeddings: Train {train_img_emb.shape}, Test {test_img_emb.shape}")
print(f"✓ Text embeddings:  Train {train_txt_emb.shape}, Test {test_txt_emb.shape}")

# Load tabular features (generated earlier)
train_df = pd.read_csv('/kaggle/input/amazon-ml-challenge-2025/dataset/train_with_features.csv')
test_df = pd.read_csv('/kaggle/input/amazon-ml-challenge-2025/dataset/test_with_features.csv')

# Define the 30 engineered features
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

print(f"✓ Tabular features: Train {X_tab_train.shape}, Test {X_tab_test.shape}")
print(f"✓ Target price range: [{y_train.min():.2f}, {y_train.max():.2f}]")

# Apply sqrt transform to target (reduces impact of outliers)
y_train_sqrt = np.sqrt(y_train)
print(f"✓ Sqrt-transformed target range: [{y_train_sqrt.min():.2f}, {y_train_sqrt.max():.2f}]")

# ============================================================================
# SECTION 2: CREATE TRAIN/VAL SPLIT
# ============================================================================
print("\n[STEP 2/8] Creating train/validation split (80/20)...")

# Random split with fixed seed for reproducibility
indices = np.arange(len(y_train))
train_idx, val_idx = train_test_split(
    indices, 
    test_size=0.2, 
    random_state=42
)

print(f"✓ Train samples: {len(train_idx):,}")
print(f"✓ Val samples:   {len(val_idx):,}")

# Split all modalities consistently
X_img_train_split, X_img_val = train_img_emb[train_idx], train_img_emb[val_idx]
X_txt_train_split, X_txt_val = train_txt_emb[train_idx], train_txt_emb[val_idx]
X_tab_train_split, X_tab_val = X_tab_train[train_idx], X_tab_train[val_idx]
y_train_split, y_val = y_train_sqrt[train_idx], y_train_sqrt[val_idx]

# ============================================================================
# SECTION 3: DEFINE MLP ARCHITECTURE
# ============================================================================
print("\n[STEP 3/8] Defining MLP architecture...")

class EmbeddingMLP(nn.Module):
    """
    MLP for CLIP embeddings (512-dim) → price prediction
    Architecture: 512 → 256 ReLU Dropout → 128 ReLU Dropout → 1
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
        
        # Output layer (no activation - regression)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights with Xavier/Glorot
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        return self.network(x).squeeze(-1)

print("✓ MLP Architecture: [512 → 256 ReLU Drop(0.3) → 128 ReLU Drop(0.2) → 1]")
print("✓ Total params per MLP: ~198K")

# ============================================================================
# SECTION 4: TRAIN IMAGE MLP (BRANCH 3 - MOST IMPORTANT)
# ============================================================================
print("\n[STEP 4/8] Training Image MLP (Branch 3 - Visual Features)...")
print("This branch learns: packaging quality, brand logos, premium aesthetics")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✓ Using device: {device}")

# Create PyTorch datasets
train_dataset_img = TensorDataset(
    torch.FloatTensor(X_img_train_split),
    torch.FloatTensor(y_train_split)
)
val_dataset_img = TensorDataset(
    torch.FloatTensor(X_img_val),
    torch.FloatTensor(y_val)
)

train_loader_img = DataLoader(train_dataset_img, batch_size=256, shuffle=True, num_workers=2, pin_memory=True)
val_loader_img = DataLoader(val_dataset_img, batch_size=512, num_workers=2, pin_memory=True)

# Initialize model
image_mlp = EmbeddingMLP().to(device)
optimizer_img = optim.AdamW(image_mlp.parameters(), lr=0.001, weight_decay=1e-5)
scheduler_img = optim.lr_scheduler.ReduceLROnPlateau(optimizer_img, mode='min', factor=0.5, patience=5, verbose=True)
criterion = nn.MSELoss()

# Training loop with early stopping
best_val_loss = float('inf')
patience = 15
patience_counter = 0
n_epochs = 100

print("Training... (early stopping patience=15)")
for epoch in range(n_epochs):
    # Train phase
    image_mlp.train()
    train_loss = 0
    for X_batch, y_batch in train_loader_img:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer_img.zero_grad()
        pred = image_mlp(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(image_mlp.parameters(), max_norm=1.0)
        
        optimizer_img.step()
        train_loss += loss.item() * len(X_batch)
    
    train_loss /= len(train_dataset_img)
    
    # Validation phase
    image_mlp.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader_img:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            pred = image_mlp(X_batch)
            loss = criterion(pred, y_batch)
            val_loss += loss.item() * len(X_batch)
    
    val_loss /= len(val_dataset_img)
    
    # Learning rate scheduling
    scheduler_img.step(val_loss)
    
    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(image_mlp.state_dict(), 'image_mlp_best.pth')
        patience_counter = 0
    else:
        patience_counter += 1
    
    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Best: {best_val_loss:.4f}")
    
    if patience_counter >= patience:
        print(f"✓ Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
        break

# Load best model
image_mlp.load_state_dict(torch.load('image_mlp_best.pth'))
print(f"✓ Best validation MSE: {best_val_loss:.4f}")

# Calculate SMAPE on validation set
image_mlp.eval()
with torch.no_grad():
    val_pred_img = image_mlp(torch.FloatTensor(X_img_val).to(device)).cpu().numpy()

# Reverse sqrt transform
val_pred_img_price = val_pred_img ** 2
y_val_price = y_val ** 2

smape_img = 100 * np.mean(2 * np.abs(val_pred_img_price - y_val_price) / (np.abs(val_pred_img_price) + np.abs(y_val_price)))
print(f"✓ Image MLP Validation SMAPE: {smape_img:.2f}%")

# ============================================================================
# SECTION 5: TRAIN TEXT MLP (BRANCH 2)
# ============================================================================
print("\n[STEP 5/8] Training Text MLP (Branch 2 - Textual Features)...")
print("This branch learns: product descriptions, keywords, specifications")

# Create PyTorch datasets
train_dataset_txt = TensorDataset(
    torch.FloatTensor(X_txt_train_split),
    torch.FloatTensor(y_train_split)
)
val_dataset_txt = TensorDataset(
    torch.FloatTensor(X_txt_val),
    torch.FloatTensor(y_val)
)

train_loader_txt = DataLoader(train_dataset_txt, batch_size=256, shuffle=True, num_workers=2, pin_memory=True)
val_loader_txt = DataLoader(val_dataset_txt, batch_size=512, num_workers=2, pin_memory=True)

# Initialize model
text_mlp = EmbeddingMLP().to(device)
optimizer_txt = optim.AdamW(text_mlp.parameters(), lr=0.001, weight_decay=1e-5)
scheduler_txt = optim.lr_scheduler.ReduceLROnPlateau(optimizer_txt, mode='min', factor=0.5, patience=5, verbose=True)

# Training loop
best_val_loss = float('inf')
patience_counter = 0

print("Training... (early stopping patience=15)")
for epoch in range(n_epochs):
    # Train phase
    text_mlp.train()
    train_loss = 0
    for X_batch, y_batch in train_loader_txt:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer_txt.zero_grad()
        pred = text_mlp(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(text_mlp.parameters(), max_norm=1.0)
        
        optimizer_txt.step()
        train_loss += loss.item() * len(X_batch)
    
    train_loss /= len(train_dataset_txt)
    
    # Validation phase
    text_mlp.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader_txt:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            pred = text_mlp(X_batch)
            loss = criterion(pred, y_batch)
            val_loss += loss.item() * len(X_batch)
    
    val_loss /= len(val_dataset_txt)
    
    scheduler_txt.step(val_loss)
    
    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(text_mlp.state_dict(), 'text_mlp_best.pth')
        patience_counter = 0
    else:
        patience_counter += 1
    
    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Best: {best_val_loss:.4f}")
    
    if patience_counter >= patience:
        print(f"✓ Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
        break

# Load best model
text_mlp.load_state_dict(torch.load('text_mlp_best.pth'))
print(f"✓ Best validation MSE: {best_val_loss:.4f}")

# Calculate SMAPE on validation
text_mlp.eval()
with torch.no_grad():
    val_pred_txt = text_mlp(torch.FloatTensor(X_txt_val).to(device)).cpu().numpy()

val_pred_txt_price = val_pred_txt ** 2
smape_txt = 100 * np.mean(2 * np.abs(val_pred_txt_price - y_val_price) / (np.abs(val_pred_txt_price) + np.abs(y_val_price)))
print(f"✓ Text MLP Validation SMAPE: {smape_txt:.2f}%")

# ============================================================================
# SECTION 6: TRAIN TABULAR LIGHTGBM (BRANCH 1)
# ============================================================================
print("\n[STEP 6/8] Training Tabular LightGBM (Branch 1 - Structured Features)...")
print("This branch learns: IPQ values, units, quality keywords, pack sizes")

# Create LightGBM datasets
lgb_train = lgb.Dataset(X_tab_train_split, y_train_split)
lgb_val = lgb.Dataset(X_tab_val, y_val, reference=lgb_train)

# Hyperparameters (tuned from baseline experiments)
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
    'random_state': 42,
    'n_jobs': -1
}

print("Training with early stopping (rounds=50)...")
lgb_model = lgb.train(
    params,
    lgb_train,
    num_boost_round=1000,
    valid_sets=[lgb_train, lgb_val],
    valid_names=['train', 'val'],
    callbacks=[
        lgb.early_stopping(50, verbose=False),
        lgb.log_evaluation(100)
    ]
)

print(f"✓ Best iteration: {lgb_model.best_iteration}")
print(f"✓ Best validation RMSE: {lgb_model.best_score['val']['rmse']:.4f}")

# Predict on validation
val_pred_tab = lgb_model.predict(X_tab_val, num_iteration=lgb_model.best_iteration)
val_pred_tab_price = val_pred_tab ** 2
smape_tab = 100 * np.mean(2 * np.abs(val_pred_tab_price - y_val_price) / (np.abs(val_pred_tab_price) + np.abs(y_val_price)))
print(f"✓ Tabular LightGBM Validation SMAPE: {smape_tab:.2f}%")

# ============================================================================
# SECTION 7: OPTIMIZE ENSEMBLE WEIGHTS
# ============================================================================
print("\n[STEP 7/8] Optimizing ensemble weights...")
print("\n📊 Individual Branch Performance:")
print(f"  Branch 1 (Tabular LGBM): {smape_tab:.2f}% SMAPE")
print(f"  Branch 2 (Text MLP):     {smape_txt:.2f}% SMAPE")
print(f"  Branch 3 (Image MLP):    {smape_img:.2f}% SMAPE")
print(f"  Best single model:       {min(smape_tab, smape_txt, smape_img):.2f}% SMAPE")

def ensemble_smape(weights, pred_tab, pred_txt, pred_img, y_true):
    """Calculate SMAPE for weighted ensemble"""
    w_tab, w_txt, w_img = weights
    ensemble_pred = w_tab * pred_tab + w_txt * pred_txt + w_img * pred_img
    smape = 100 * np.mean(2 * np.abs(ensemble_pred - y_true) / (np.abs(ensemble_pred) + np.abs(y_true)))
    return smape

# Optimize weights using scipy
def objective(weights):
    return ensemble_smape(weights, val_pred_tab_price, val_pred_txt_price, val_pred_img_price, y_val_price)

# Constraints: weights sum to 1, all non-negative
constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
bounds = [(0, 1), (0, 1), (0, 1)]
initial_weights = [0.2, 0.3, 0.5]  # Educated guess: image > text > tabular

print(f"\nStarting optimization with initial weights: [tab={initial_weights[0]}, txt={initial_weights[1]}, img={initial_weights[2]}]")
print(f"Initial ensemble SMAPE: {objective(initial_weights):.2f}%")

result = minimize(
    objective,
    initial_weights,
    method='SLSQP',
    bounds=bounds,
    constraints=constraints,
    options={'maxiter': 100}
)

optimal_weights = result.x
ensemble_smape_final = result.fun

print(f"\n✓ Optimization complete!")
print(f"✓ Optimal weights: [tab={optimal_weights[0]:.3f}, txt={optimal_weights[1]:.3f}, img={optimal_weights[2]:.3f}]")
print(f"✓ Ensemble Validation SMAPE: {ensemble_smape_final:.2f}%")

improvement = min(smape_tab, smape_txt, smape_img) - ensemble_smape_final
print(f"\n📈 Improvement over best single model: {improvement:.2f}% SMAPE")

if improvement > 0:
    print("   ✅ Ensemble is better than individual models!")
else:
    print("   ⚠️  Ensemble not better - will use best single model")
    # Fallback to best single model
    best_idx = np.argmin([smape_tab, smape_txt, smape_img])
    optimal_weights = [[1, 0, 0], [0, 1, 0], [0, 0, 1]][best_idx]
    print(f"   Using best single model weights: {optimal_weights}")

# ============================================================================
# SECTION 8: GENERATE TEST PREDICTIONS
# ============================================================================
print("\n[STEP 8/8] Generating final test predictions...")

# Retrain all models on FULL training set for final predictions
print("Re-training all models on full training data...")

# --- Image MLP on full train ---
print("  [1/3] Image MLP...")
train_dataset_img_full = TensorDataset(
    torch.FloatTensor(train_img_emb),
    torch.FloatTensor(y_train_sqrt)
)
train_loader_img_full = DataLoader(train_dataset_img_full, batch_size=256, shuffle=True, num_workers=2)

image_mlp_final = EmbeddingMLP().to(device)
optimizer_img_final = optim.AdamW(image_mlp_final.parameters(), lr=0.001, weight_decay=1e-5)

# Train for fewer epochs (we already validated hyperparameters)
for epoch in range(50):
    image_mlp_final.train()
    for X_batch, y_batch in train_loader_img_full:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer_img_final.zero_grad()
        pred = image_mlp_final(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(image_mlp_final.parameters(), max_norm=1.0)
        optimizer_img_final.step()

print("        ✓ Image MLP trained on 75k samples")

# --- Text MLP on full train ---
print("  [2/3] Text MLP...")
train_dataset_txt_full = TensorDataset(
    torch.FloatTensor(train_txt_emb),
    torch.FloatTensor(y_train_sqrt)
)
train_loader_txt_full = DataLoader(train_dataset_txt_full, batch_size=256, shuffle=True, num_workers=2)

text_mlp_final = EmbeddingMLP().to(device)
optimizer_txt_final = optim.AdamW(text_mlp_final.parameters(), lr=0.001, weight_decay=1e-5)

for epoch in range(50):
    text_mlp_final.train()
    for X_batch, y_batch in train_loader_txt_full:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer_txt_final.zero_grad()
        pred = text_mlp_final(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(text_mlp_final.parameters(), max_norm=1.0)
        optimizer_txt_final.step()

print("        ✓ Text MLP trained on 75k samples")

# --- LightGBM on full train ---
print("  [3/3] Tabular LightGBM...")
lgb_train_full = lgb.Dataset(X_tab_train, y_train_sqrt)
lgb_model_final = lgb.train(
    params,
    lgb_train_full,
    num_boost_round=lgb_model.best_iteration,
    callbacks=[lgb.log_evaluation(0)]  # Silent
)
print("        ✓ LightGBM trained on 75k samples")

# Generate test predictions from all branches
print("\nGenerating test predictions...")

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

print(f"✓ Tabular predictions range: [{test_pred_tab_price.min():.2f}, {test_pred_tab_price.max():.2f}]")
print(f"✓ Text predictions range:    [{test_pred_txt_price.min():.2f}, {test_pred_txt_price.max():.2f}]")
print(f"✓ Image predictions range:   [{test_pred_img_price.min():.2f}, {test_pred_img_price.max():.2f}]")

# Apply optimal ensemble weights
test_pred_ensemble = (
    optimal_weights[0] * test_pred_tab_price +
    optimal_weights[1] * test_pred_txt_price +
    optimal_weights[2] * test_pred_img_price
)

# Post-processing: clip to reasonable range
test_pred_ensemble = np.clip(test_pred_ensemble, 0.01, None)  # No negative prices

print(f"✓ Ensemble predictions range: [{test_pred_ensemble.min():.2f}, {test_pred_ensemble.max():.2f}]")

# ============================================================================
# SAVE SUBMISSION
# ============================================================================
submission = pd.DataFrame({
    'sample_id': test_df['sample_id'],
    'price': test_pred_ensemble
})

submission.to_csv('submission_multimodal_ensemble.csv', index=False)

print("\n" + "=" * 80)
print("✅ MULTI-MODAL ENSEMBLE TRAINING COMPLETE!")
print("=" * 80)
print(f"\n📊 FINAL RESULTS:")
print(f"   Validation SMAPE:    {ensemble_smape_final:.2f}%")
print(f"   Ensemble weights:    Tab={optimal_weights[0]:.3f}, Txt={optimal_weights[1]:.3f}, Img={optimal_weights[2]:.3f}")
print(f"   Submission file:     submission_multimodal_ensemble.csv")
print(f"   Test samples:        {len(submission):,}")
print(f"\n📝 Sample predictions:")
print(submission.head(10))
print(f"\n🎯 Next steps:")
print(f"   1. Download submission_multimodal_ensemble.csv")
print(f"   2. Submit to Kaggle leaderboard")
print(f"   3. Compare to baseline (62.45% SMAPE)")
print(f"   4. If good results, proceed to Option D (CLIP fine-tuning)")
print("=" * 80)
