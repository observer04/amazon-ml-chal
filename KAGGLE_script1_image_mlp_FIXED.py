"""
FIXED DIAGNOSTIC SCRIPT 1: IMAGE MLP TRAINING
==============================================
MAJOR FIXES:
1. ✅ log(price) transform instead of sqrt (handles wide price range better)
2. ✅ Better architecture and training (addresses underprediction)
3. ✅ More epochs and patience (prevents premature stopping)

Previous issues:
- 75.53% SMAPE (TERRIBLE)
- Predictions collapsed to low values
- SMAPE > 100% for cheap and expensive items

Expected improvement: 75% → 35-45% SMAPE

Time: 30-40 minutes
GPU: Required
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("FIXED DIAGNOSTIC 1: IMAGE MLP TRAINING (with log transform)")
print("Objective: Evaluate if CLIP image embeddings predict price")
print("=" * 80)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n[1/7] Loading image embeddings...")

train_img_emb = np.load('/kaggle/working/amazon-ml-chal/outputs/train_image_embeddings_clip.npy')
test_img_emb = np.load('/kaggle/working/amazon-ml-chal/outputs/test_image_embeddings_clip.npy')

train_df = pd.read_csv('/kaggle/working/amazon-ml-chal/dataset/train_with_features.csv')
y_train = train_df['price'].values

print(f"✓ Image embeddings: {train_img_emb.shape}")
print(f"✓ Price range: [{y_train.min():.2f}, {y_train.max():.2f}]")
print(f"✓ Price mean: {y_train.mean():.2f}, median: {np.median(y_train):.2f}")

# 🔧 FIX 1: Use log transform instead of sqrt
y_train_log = np.log1p(y_train)  # log(1+x) handles zeros
print(f"✓ Log-transformed target range: [{y_train_log.min():.2f}, {y_train_log.max():.2f}]")

# ============================================================================
# TRAIN/VAL SPLIT
# ============================================================================
print("\n[2/7] Creating train/validation split (80/20)...")

indices = np.arange(len(y_train))
train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

X_train, X_val = train_img_emb[train_idx], train_img_emb[val_idx]
y_train_split, y_val = y_train_log[train_idx], y_train_log[val_idx]

print(f"✓ Train: {len(train_idx):,} samples")
print(f"✓ Val:   {len(val_idx):,} samples")

# ============================================================================
# DEFINE MODEL
# ============================================================================
print("\n[3/7] Defining MLP architecture...")

class ImageMLP(nn.Module):
    """
    Improved MLP for image embeddings: 512 → 256 → 128 → 64 → 1
    Deeper network with BatchNorm for better training
    """
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(64, 1)
        )
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.network(x).squeeze(-1)

model = ImageMLP()
total_params = sum(p.numel() for p in model.parameters())
print(f"✓ Architecture: [512 → 256 BN → 128 BN → 64 BN → 1]")
print(f"✓ Total parameters: {total_params:,}")

# ============================================================================
# SETUP TRAINING
# ============================================================================
print("\n[4/7] Setting up training...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✓ Device: {device}")

model = model.to(device)

# Create dataloaders
train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train_split))
val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=512, num_workers=2, pin_memory=True)

# 🔧 FIX 2: Better optimizer settings
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
criterion = nn.MSELoss()

print(f"✓ Optimizer: AdamW (lr=0.001, weight_decay=1e-4)")
print(f"✓ Scheduler: ReduceLROnPlateau (factor=0.5, patience=10)")
print(f"✓ Batch size: 256 (train), 512 (val)")
print(f"✓ Early stopping: patience=20 (increased from 15)")

# ============================================================================
# TRAINING LOOP WITH DIAGNOSTICS
# ============================================================================
print("\n[5/7] Training with detailed logging...")
print("\nEpoch | Train Loss | Val Loss | Val SMAPE | LR      | Status")
print("-" * 70)

# Tracking
history = {
    'train_loss': [],
    'val_loss': [],
    'val_smape': [],
    'lr': []
}

best_val_loss = float('inf')
best_val_smape = float('inf')
patience = 20  # Increased patience
patience_counter = 0
n_epochs = 150  # More epochs

for epoch in range(n_epochs):
    # === TRAIN PHASE ===
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        train_loss += loss.item() * len(X_batch)
    
    train_loss /= len(train_dataset)
    
    # === VALIDATION PHASE ===
    model.eval()
    val_loss = 0
    val_preds = []
    val_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            val_loss += loss.item() * len(X_batch)
            
            val_preds.append(pred.cpu().numpy())
            val_targets.append(y_batch.cpu().numpy())
    
    val_loss /= len(val_dataset)
    val_preds = np.concatenate(val_preds)
    val_targets = np.concatenate(val_targets)
    
    # 🔧 FIX 3: Reverse log transform properly
    val_preds_price = np.expm1(val_preds)  # exp(x) - 1
    val_targets_price = np.expm1(val_targets)
    
    # Calculate SMAPE
    val_smape = 100 * np.mean(
        2 * np.abs(val_preds_price - val_targets_price) / 
        (np.abs(val_preds_price) + np.abs(val_targets_price))
    )
    
    # Learning rate
    current_lr = optimizer.param_groups[0]['lr']
    
    # Save history
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['val_smape'].append(val_smape)
    history['lr'].append(current_lr)
    
    # Learning rate scheduling
    scheduler.step(val_loss)
    
    # Early stopping check
    status = ""
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_val_smape = val_smape
        torch.save(model.state_dict(), 'image_mlp_best_FIXED.pth')
        patience_counter = 0
        status = "✓ BEST"
    else:
        patience_counter += 1
        status = f"({patience_counter}/{patience})"
    
    # Print progress every 5 epochs
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"{epoch+1:5d} | {train_loss:10.4f} | {val_loss:8.4f} | "
              f"{val_smape:9.2f}% | {current_lr:.6f} | {status}")
    
    # Early stopping
    if patience_counter >= patience:
        print(f"\n✓ Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
        break

# Load best model
model.load_state_dict(torch.load('image_mlp_best_FIXED.pth'))
print(f"\n✓ Best validation loss: {best_val_loss:.4f}")
print(f"✓ Best validation SMAPE: {best_val_smape:.2f}%")

# ============================================================================
# DIAGNOSTIC PLOTS
# ============================================================================
print("\n[6/7] Generating diagnostic plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Training curves
ax = axes[0, 0]
ax.plot(history['train_loss'], label='Train Loss', linewidth=2)
ax.plot(history['val_loss'], label='Val Loss', linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE Loss')
ax.set_title('Training Curves: Loss')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: SMAPE over epochs
ax = axes[0, 1]
ax.plot(history['val_smape'], color='green', linewidth=2)
ax.axhline(y=best_val_smape, color='red', linestyle='--', label=f'Best: {best_val_smape:.2f}%')
ax.set_xlabel('Epoch')
ax.set_ylabel('SMAPE (%)')
ax.set_title('Validation SMAPE Over Time')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Prediction distribution
model.eval()
with torch.no_grad():
    final_val_preds = model(torch.FloatTensor(X_val).to(device)).cpu().numpy()
final_val_preds_price = np.expm1(final_val_preds)
final_val_targets_price = np.expm1(y_val)

ax = axes[1, 0]
ax.hist(final_val_targets_price, bins=50, alpha=0.5, label='Actual', color='blue')
ax.hist(final_val_preds_price, bins=50, alpha=0.5, label='Predicted', color='orange')
ax.set_xlabel('Price ($)')
ax.set_ylabel('Frequency')
ax.set_title('Prediction Distribution vs Actual')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Actual vs Predicted scatter
ax = axes[1, 1]
ax.scatter(final_val_targets_price, final_val_preds_price, alpha=0.3, s=10)
ax.plot([0, final_val_targets_price.max()], [0, final_val_targets_price.max()], 
        'r--', linewidth=2, label='Perfect prediction')
ax.set_xlabel('Actual Price ($)')
ax.set_ylabel('Predicted Price ($)')
ax.set_title('Actual vs Predicted (Validation Set)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('image_mlp_diagnostics_FIXED.png', dpi=150, bbox_inches='tight')
print("✓ Saved diagnostic plots to: image_mlp_diagnostics_FIXED.png")

# ============================================================================
# DETAILED ANALYSIS
# ============================================================================
print("\n[7/7] Detailed performance analysis...")

# Error statistics
errors = np.abs(final_val_preds_price - final_val_targets_price)
relative_errors = errors / (final_val_targets_price + 1e-8)

print(f"\n📊 Error Statistics:")
print(f"   Mean Absolute Error (MAE):  ${np.mean(errors):.2f}")
print(f"   Median Absolute Error:       ${np.median(errors):.2f}")
print(f"   90th percentile error:       ${np.percentile(errors, 90):.2f}")
print(f"   Max error:                   ${np.max(errors):.2f}")
print(f"\n   Mean Relative Error:         {np.mean(relative_errors)*100:.2f}%")
print(f"   Median Relative Error:       {np.median(relative_errors)*100:.2f}%")

# Predictions by price range
print(f"\n📈 Performance by Price Range:")
ranges = [(0, 10), (10, 25), (25, 50), (50, 100), (100, 1000), (1000, 10000)]
for low, high in ranges:
    mask = (final_val_targets_price >= low) & (final_val_targets_price < high)
    if mask.sum() > 0:
        range_smape = 100 * np.mean(
            2 * np.abs(final_val_preds_price[mask] - final_val_targets_price[mask]) / 
            (np.abs(final_val_preds_price[mask]) + np.abs(final_val_targets_price[mask]))
        )
        print(f"   ${low:4d}-${high:5d}: {range_smape:6.2f}% SMAPE ({mask.sum():5,} samples)")

# ============================================================================
# RECOMMENDATIONS
# ============================================================================
print("\n" + "=" * 80)
print("📋 DIAGNOSTIC SUMMARY - FIXED VERSION")
print("=" * 80)

print(f"\n✅ Model Training Complete:")
print(f"   - Best Validation SMAPE: {best_val_smape:.2f}%")
print(f"   - Converged at epoch: {len(history['train_loss'])}")
print(f"   - Final learning rate: {history['lr'][-1]:.6f}")

print(f"\n🎯 Interpretation:")
if best_val_smape < 35:
    print(f"   ✅ EXCELLENT: Image embeddings are highly predictive!")
    print(f"      SMAPE < 35% means visual features capture pricing very well.")
    recommendation = "PROCEED to Text MLP training - looking great!"
elif best_val_smape < 45:
    print(f"   ✅ GOOD: Image embeddings are useful for price prediction.")
    print(f"      SMAPE < 45% indicates strong visual pricing cues.")
    recommendation = "PROCEED to Text MLP training"
elif best_val_smape < 55:
    print(f"   ⚠️  MODERATE: Image embeddings provide some signal.")
    print(f"      SMAPE < 55% suggests limited but positive contribution.")
    recommendation = "PROCEED - ensemble may help compensate"
else:
    print(f"   ❌ WEAK: Still not good enough even with log transform.")
    print(f"      SMAPE > 55% indicates fundamental issues.")
    recommendation = "INVESTIGATE: Check embeddings quality, consider different architecture"

print(f"\n💡 Recommendation: {recommendation}")

print(f"\n📁 Saved Files:")
print(f"   - image_mlp_best_FIXED.pth (model weights)")
print(f"   - image_mlp_diagnostics_FIXED.png (plots)")

print(f"\n🔧 Improvements Made:")
print(f"   1. log(price) instead of sqrt(price) - handles wide range better")
print(f"   2. Deeper network (4 layers) with BatchNorm - more capacity")
print(f"   3. Better optimizer settings - more stable training")
print(f"   4. More epochs (150) and patience (20) - avoids early stopping")

print(f"\n🚀 Next Steps:")
print(f"   1. Review diagnostic plots (image_mlp_diagnostics_FIXED.png)")
print(f"   2. If satisfied, run KAGGLE_script2_text_mlp_FIXED.py")
print(f"   3. Compare Image vs Text MLP performance")
print("=" * 80)
