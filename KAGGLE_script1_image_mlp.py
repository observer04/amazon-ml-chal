"""
DIAGNOSTIC SCRIPT 1: IMAGE MLP TRAINING
========================================
This script trains an MLP on CLIP image embeddings (512-dim) 
and provides detailed diagnostics to evaluate if visual features
are useful for price prediction.

What to watch:
- Training should converge smoothly (loss decreasing)
- Validation SMAPE < 50% indicates image features are useful
- Prediction distribution should match actual price distribution
- No severe overfitting (train/val gap should be small)

Time: 20-30 minutes
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
print("DIAGNOSTIC 1: IMAGE MLP TRAINING")
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

# Apply sqrt transform (helps with skewed distribution)
y_train_sqrt = np.sqrt(y_train)
print(f"✓ Sqrt-transformed target range: [{y_train_sqrt.min():.2f}, {y_train_sqrt.max():.2f}]")

# ============================================================================
# TRAIN/VAL SPLIT
# ============================================================================
print("\n[2/7] Creating train/validation split (80/20)...")

indices = np.arange(len(y_train))
train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

X_train, X_val = train_img_emb[train_idx], train_img_emb[val_idx]
y_train_split, y_val = y_train_sqrt[train_idx], y_train_sqrt[val_idx]

print(f"✓ Train: {len(train_idx):,} samples")
print(f"✓ Val:   {len(val_idx):,} samples")

# ============================================================================
# DEFINE MODEL
# ============================================================================
print("\n[3/7] Defining MLP architecture...")

class ImageMLP(nn.Module):
    """
    MLP for image embeddings: 512 → 256 → 128 → 1
    Uses ReLU activation, Dropout for regularization
    """
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
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
print(f"✓ Architecture: [512 → 256 ReLU Drop(0.3) → 128 ReLU Drop(0.2) → 1]")
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

# Optimizer and scheduler
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
criterion = nn.MSELoss()

print(f"✓ Optimizer: AdamW (lr=0.001, weight_decay=1e-5)")
print(f"✓ Scheduler: ReduceLROnPlateau (factor=0.5, patience=5)")
print(f"✓ Batch size: 256 (train), 512 (val)")
print(f"✓ Early stopping: patience=15")

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
patience = 15
patience_counter = 0
n_epochs = 100

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
    
    # Calculate SMAPE (reverse sqrt transform first)
    val_preds_price = val_preds ** 2
    val_targets_price = val_targets ** 2
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
        torch.save(model.state_dict(), 'image_mlp_best.pth')
        patience_counter = 0
        status = "✓ BEST"
    else:
        patience_counter += 1
        status = f"({patience_counter}/{patience})"
    
    # Print progress
    print(f"{epoch+1:5d} | {train_loss:10.4f} | {val_loss:8.4f} | "
          f"{val_smape:9.2f}% | {current_lr:.6f} | {status}")
    
    # Early stopping
    if patience_counter >= patience:
        print(f"\n✓ Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
        break

# Load best model
model.load_state_dict(torch.load('image_mlp_best.pth'))
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
final_val_preds_price = final_val_preds ** 2
final_val_targets_price = y_val ** 2

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
plt.savefig('image_mlp_diagnostics.png', dpi=150, bbox_inches='tight')
print("✓ Saved diagnostic plots to: image_mlp_diagnostics.png")

# ============================================================================
# DETAILED ANALYSIS
# ============================================================================
print("\n[7/7] Detailed performance analysis...")

# Error statistics
errors = np.abs(final_val_preds_price - final_val_targets_price)
relative_errors = errors / final_val_targets_price

print(f"\n📊 Error Statistics:")
print(f"   Mean Absolute Error (MAE):  ${np.mean(errors):.2f}")
print(f"   Median Absolute Error:       ${np.median(errors):.2f}")
print(f"   90th percentile error:       ${np.percentile(errors, 90):.2f}")
print(f"   Max error:                   ${np.max(errors):.2f}")
print(f"\n   Mean Relative Error:         {np.mean(relative_errors)*100:.2f}%")
print(f"   Median Relative Error:       {np.median(relative_errors)*100:.2f}%")

# Predictions by price range
print(f"\n📈 Performance by Price Range:")
ranges = [(0, 10), (10, 25), (25, 50), (50, 100), (100, 1000)]
for low, high in ranges:
    mask = (final_val_targets_price >= low) & (final_val_targets_price < high)
    if mask.sum() > 0:
        range_smape = 100 * np.mean(
            2 * np.abs(final_val_preds_price[mask] - final_val_targets_price[mask]) / 
            (np.abs(final_val_preds_price[mask]) + np.abs(final_val_targets_price[mask]))
        )
        print(f"   ${low:3d}-${high:3d}: {range_smape:6.2f}% SMAPE ({mask.sum():5,} samples)")

# ============================================================================
# RECOMMENDATIONS
# ============================================================================
print("\n" + "=" * 80)
print("📋 DIAGNOSTIC SUMMARY")
print("=" * 80)

print(f"\n✅ Model Training Complete:")
print(f"   - Best Validation SMAPE: {best_val_smape:.2f}%")
print(f"   - Converged at epoch: {len(history['train_loss'])}")
print(f"   - Final learning rate: {history['lr'][-1]:.6f}")

print(f"\n🎯 Interpretation:")
if best_val_smape < 40:
    print(f"   ✅ EXCELLENT: Image embeddings are highly predictive!")
    print(f"      SMAPE < 40% means visual features capture pricing well.")
    recommendation = "PROCEED to Text MLP training"
elif best_val_smape < 50:
    print(f"   ✅ GOOD: Image embeddings are useful for price prediction.")
    print(f"      SMAPE < 50% indicates visual cues help.")
    recommendation = "PROCEED to Text MLP training"
elif best_val_smape < 60:
    print(f"   ⚠️  MODERATE: Image embeddings provide some signal.")
    print(f"      SMAPE < 60% suggests limited but positive contribution.")
    recommendation = "PROCEED with caution - may need hyperparameter tuning"
else:
    print(f"   ❌ WEAK: Image embeddings don't predict price well alone.")
    print(f"      SMAPE > 60% indicates visual features insufficient.")
    recommendation = "INVESTIGATE: Check embeddings quality, try different transform"

print(f"\n💡 Recommendation: {recommendation}")

print(f"\n📁 Saved Files:")
print(f"   - image_mlp_best.pth (model weights)")
print(f"   - image_mlp_diagnostics.png (plots)")

print(f"\n🚀 Next Steps:")
print(f"   1. Review diagnostic plots (image_mlp_diagnostics.png)")
print(f"   2. If satisfied, run KAGGLE_script2_text_mlp.py")
print(f"   3. Compare Image vs Text MLP performance")
print("=" * 80)
