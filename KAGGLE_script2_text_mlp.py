"""
DIAGNOSTIC SCRIPT 2: TEXT MLP TRAINING
=======================================
This script trains an MLP on CLIP text embeddings (512-dim) 
and compares performance to Image MLP.

What to watch:
- Compare Text SMAPE to Image SMAPE from Script 1
- Check if text provides complementary or redundant information
- Validation SMAPE < 55% indicates text features are useful

Time: 20-30 minutes
GPU: Required

IMPORTANT: Run KAGGLE_script1_image_mlp.py FIRST!
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
print("DIAGNOSTIC 2: TEXT MLP TRAINING")
print("Objective: Evaluate if CLIP text embeddings predict price")
print("=" * 80)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n[1/7] Loading text embeddings...")

train_txt_emb = np.load('/kaggle/working/amazon-ml-chal/outputs/train_text_embeddings_clip.npy')
test_txt_emb = np.load('/kaggle/working/amazon-ml-chal/outputs/test_text_embeddings_clip.npy')

train_df = pd.read_csv('/kaggle/working/amazon-ml-chal/dataset/train_with_features.csv')
y_train = train_df['price'].values

print(f"✓ Text embeddings: {train_txt_emb.shape}")
print(f"✓ Price range: [{y_train.min():.2f}, {y_train.max():.2f}]")

y_train_sqrt = np.sqrt(y_train)

# ============================================================================
# TRAIN/VAL SPLIT (SAME AS IMAGE MLP)
# ============================================================================
print("\n[2/7] Creating train/validation split (same seed as Image MLP)...")

indices = np.arange(len(y_train))
train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

X_train, X_val = train_txt_emb[train_idx], train_txt_emb[val_idx]
y_train_split, y_val = y_train_sqrt[train_idx], y_train_sqrt[val_idx]

print(f"✓ Train: {len(train_idx):,} samples")
print(f"✓ Val:   {len(val_idx):,} samples")

# ============================================================================
# DEFINE MODEL (SAME ARCHITECTURE AS IMAGE MLP)
# ============================================================================
print("\n[3/7] Defining MLP architecture...")

class TextMLP(nn.Module):
    """Same architecture as Image MLP for fair comparison"""
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

model = TextMLP()
print(f"✓ Architecture: [512 → 256 ReLU Drop(0.3) → 128 ReLU Drop(0.2) → 1]")

# ============================================================================
# SETUP TRAINING
# ============================================================================
print("\n[4/7] Setting up training...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train_split))
val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=512, num_workers=2, pin_memory=True)

optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
criterion = nn.MSELoss()

print(f"✓ Same hyperparameters as Image MLP")

# ============================================================================
# TRAINING LOOP
# ============================================================================
print("\n[5/7] Training with detailed logging...")
print("\nEpoch | Train Loss | Val Loss | Val SMAPE | LR      | Status")
print("-" * 70)

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
    # Train
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item() * len(X_batch)
    train_loss /= len(train_dataset)
    
    # Validate
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
    
    # SMAPE
    val_preds_price = val_preds ** 2
    val_targets_price = val_targets ** 2
    val_smape = 100 * np.mean(
        2 * np.abs(val_preds_price - val_targets_price) / 
        (np.abs(val_preds_price) + np.abs(val_targets_price))
    )
    
    current_lr = optimizer.param_groups[0]['lr']
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['val_smape'].append(val_smape)
    history['lr'].append(current_lr)
    
    scheduler.step(val_loss)
    
    status = ""
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_val_smape = val_smape
        torch.save(model.state_dict(), 'text_mlp_best.pth')
        patience_counter = 0
        status = "✓ BEST"
    else:
        patience_counter += 1
        status = f"({patience_counter}/{patience})"
    
    print(f"{epoch+1:5d} | {train_loss:10.4f} | {val_loss:8.4f} | "
          f"{val_smape:9.2f}% | {current_lr:.6f} | {status}")
    
    if patience_counter >= patience:
        print(f"\n✓ Early stopping at epoch {epoch+1}")
        break

model.load_state_dict(torch.load('text_mlp_best.pth'))
print(f"\n✓ Best validation SMAPE: {best_val_smape:.2f}%")

# ============================================================================
# DIAGNOSTIC PLOTS
# ============================================================================
print("\n[6/7] Generating diagnostic plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Training curves
ax = axes[0, 0]
ax.plot(history['train_loss'], label='Train Loss', linewidth=2)
ax.plot(history['val_loss'], label='Val Loss', linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE Loss')
ax.set_title('Training Curves: Loss')
ax.legend()
ax.grid(True, alpha=0.3)

# SMAPE
ax = axes[0, 1]
ax.plot(history['val_smape'], color='green', linewidth=2)
ax.axhline(y=best_val_smape, color='red', linestyle='--', label=f'Best: {best_val_smape:.2f}%')
ax.set_xlabel('Epoch')
ax.set_ylabel('SMAPE (%)')
ax.set_title('Validation SMAPE Over Time')
ax.legend()
ax.grid(True, alpha=0.3)

# Prediction distribution
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

# Scatter
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
plt.savefig('text_mlp_diagnostics.png', dpi=150, bbox_inches='tight')
print("✓ Saved diagnostic plots to: text_mlp_diagnostics.png")

# ============================================================================
# COMPARISON WITH IMAGE MLP
# ============================================================================
print("\n[7/7] Comparing with Image MLP...")

# Try to load image MLP results
try:
    print("\n✓ Loading Image MLP for comparison...")
    
    # Define same architecture
    class ImageMLP(nn.Module):
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
        def forward(self, x):
            return self.network(x).squeeze(-1)
    
    # Load Image MLP
    image_model = ImageMLP().to(device)
    image_model.load_state_dict(torch.load('image_mlp_best.pth'))
    image_model.eval()
    
    # Load image embeddings for validation set
    train_img_emb = np.load('/kaggle/working/amazon-ml-chal/outputs/train_image_embeddings_clip.npy')
    X_img_val = train_img_emb[val_idx]
    
    # Predict with Image MLP
    with torch.no_grad():
        img_val_preds = image_model(torch.FloatTensor(X_img_val).to(device)).cpu().numpy()
    img_val_preds_price = img_val_preds ** 2
    
    img_smape = 100 * np.mean(
        2 * np.abs(img_val_preds_price - final_val_targets_price) / 
        (np.abs(img_val_preds_price) + np.abs(final_val_targets_price))
    )
    
    print(f"\n📊 COMPARISON:")
    print(f"   Image MLP SMAPE: {img_smape:.2f}%")
    print(f"   Text MLP SMAPE:  {best_val_smape:.2f}%")
    print(f"   Difference:      {best_val_smape - img_smape:+.2f}%")
    
    if best_val_smape < img_smape:
        print(f"\n   ✅ Text MLP is BETTER than Image MLP!")
    elif best_val_smape < img_smape + 5:
        print(f"\n   ✅ Text MLP is comparable to Image MLP")
    else:
        print(f"\n   ⚠️  Image MLP outperforms Text MLP")
    
    # Check correlation
    correlation = np.corrcoef(img_val_preds_price, final_val_preds_price)[0, 1]
    print(f"\n   Prediction correlation: {correlation:.3f}")
    if correlation > 0.8:
        print(f"   ⚠️  HIGH correlation - modalities may be redundant")
    elif correlation > 0.5:
        print(f"   ✅ MODERATE correlation - some overlap but complementary")
    else:
        print(f"   ✅ LOW correlation - modalities capture different signals")

except Exception as e:
    print(f"\n⚠️  Could not load Image MLP for comparison: {e}")
    print("   This is okay if you haven't run KAGGLE_script1_image_mlp.py yet")

# ============================================================================
# RECOMMENDATIONS
# ============================================================================
print("\n" + "=" * 80)
print("📋 TEXT MLP DIAGNOSTIC SUMMARY")
print("=" * 80)

print(f"\n✅ Model Training Complete:")
print(f"   - Best Validation SMAPE: {best_val_smape:.2f}%")

print(f"\n🎯 Interpretation:")
if best_val_smape < 50:
    print(f"   ✅ GOOD: Text embeddings are useful for price prediction.")
    recommendation = "PROCEED to final ensemble"
elif best_val_smape < 60:
    print(f"   ⚠️  MODERATE: Text embeddings provide some signal.")
    recommendation = "PROCEED with caution - ensemble may help"
else:
    print(f"   ❌ WEAK: Text embeddings alone are insufficient.")
    recommendation = "RELY on Image MLP primarily, use text as minor component"

print(f"\n💡 Recommendation: {recommendation}")

print(f"\n📁 Saved Files:")
print(f"   - text_mlp_best.pth (model weights)")
print(f"   - text_mlp_diagnostics.png (plots)")

print(f"\n🚀 Next Steps:")
print(f"   1. Review and compare both diagnostic plots")
print(f"   2. Decide on ensemble strategy:")
print(f"      - If both < 50%: Equal weights likely optimal")
print(f"      - If one much better: Weight it higher")
print(f"   3. Run KAGGLE_script3_ensemble.py")
print("=" * 80)
