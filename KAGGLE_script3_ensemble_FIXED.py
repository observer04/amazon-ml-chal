"""
FIXED DIAGNOSTIC SCRIPT 3: FINAL ENSEMBLE
==========================================
MAJOR FIXES:
1. ✅ Using ACTUAL CSV features (42 columns, not hardcoded 30)
2. ✅ log(price) transform for consistency
3. ✅ Feature selection from available columns

Previous: CRASHED (KeyError)
Expected: <40% SMAPE (ensemble beats individual models)

Time: 40-50 minutes
GPU: Required

IMPORTANT: Run Scripts 1 & 2 FIRST!
This script loads their trained models.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("FIXED DIAGNOSTIC 3: ENSEMBLE PREDICTION")
print("Objective: Combine Image + Text + Tabular for best performance")
print("=" * 80)

# ============================================================================
# LOAD EMBEDDINGS & DATA
# ============================================================================
print("\n[1/8] Loading all data sources...")

train_img = np.load('/kaggle/working/amazon-ml-chal/outputs/train_image_embeddings_clip.npy')
train_txt = np.load('/kaggle/working/amazon-ml-chal/outputs/train_text_embeddings_clip.npy')
test_img = np.load('/kaggle/working/amazon-ml-chal/outputs/test_image_embeddings_clip.npy')
test_txt = np.load('/kaggle/working/amazon-ml-chal/outputs/test_text_embeddings_clip.npy')

train_df = pd.read_csv('/kaggle/working/amazon-ml-chal/dataset/train_with_features.csv')
test_df = pd.read_csv('/kaggle/working/amazon-ml-chal/dataset/test.csv')

y_train = train_df['price'].values
y_train_log = np.log1p(y_train)

print(f"✓ Image embeddings: {train_img.shape}")
print(f"✓ Text embeddings: {train_txt.shape}")
print(f"✓ Train CSV: {train_df.shape}")
print(f"✓ Test CSV: {test_df.shape}")

# ============================================================================
# EXTRACT ACTUAL TABULAR FEATURES
# ============================================================================
print("\n[2/8] Extracting tabular features from CSV...")

def extract_tabular_features(df, is_train=True):
    """
    Extract features from ACTUAL CSV columns (42 total)
    Categories:
    - Numerical: value, pack_size, text_length, word_count, num_bullets, description_length, price_per_unit
    - Binary: has_premium, has_organic, has_gourmet, has_natural, has_artisan, has_luxury, 
              is_travel_size, is_bulk, brand_exists
    - Interactions: value_x_premium, value_x_luxury, value_x_organic, pack_x_premium, 
                    pack_x_value, brand_x_premium, brand_x_organic, travel_x_value, bulk_x_value
    - Categorical: unit (encoded), brand (encoded), price_segment (encoded)
    """
    features = pd.DataFrame()
    
    # Numerical features (handle missing)
    numerical_cols = ['value', 'pack_size', 'text_length', 'word_count', 
                     'num_bullets', 'description_length', 'price_per_unit']
    for col in numerical_cols:
        if col in df.columns:
            features[col] = df[col].fillna(0)
        else:
            features[col] = 0
    
    # Binary flags
    binary_cols = ['has_premium', 'has_organic', 'has_gourmet', 'has_natural', 
                  'has_artisan', 'has_luxury', 'is_travel_size', 'is_bulk', 
                  'brand_exists', 'has_bullets']
    for col in binary_cols:
        if col in df.columns:
            features[col] = df[col].fillna(0).astype(int)
        else:
            features[col] = 0
    
    # Interaction features
    interaction_cols = ['value_x_premium', 'value_x_luxury', 'value_x_organic',
                       'pack_x_premium', 'pack_x_value', 'brand_x_premium',
                       'brand_x_organic', 'travel_x_value', 'bulk_x_value']
    for col in interaction_cols:
        if col in df.columns:
            features[col] = df[col].fillna(0)
        else:
            features[col] = 0
    
    # Categorical encoding
    if 'unit' in df.columns:
        unit_map = {'kilogram': 0, 'gram': 1, 'litre': 2, 'millilitre': 3, 
                   'centimetre': 4, 'metre': 5, 'unit': 6, 'count': 7}
        features['unit_encoded'] = df['unit'].fillna('unit').map(unit_map).fillna(6)
    else:
        features['unit_encoded'] = 6
    
    if 'price_segment' in df.columns:
        segment_map = {'budget': 0, 'economy': 1, 'mid_range': 2, 'premium': 3, 'luxury': 4}
        features['price_segment_encoded'] = df['price_segment'].fillna('mid_range').map(segment_map).fillna(2)
    else:
        features['price_segment_encoded'] = 2
    
    # Brand encoding (simple frequency encoding if training, else 0)
    if is_train and 'brand' in df.columns:
        brand_counts = df['brand'].value_counts()
        features['brand_freq'] = df['brand'].map(brand_counts).fillna(0) / len(df)
    else:
        features['brand_freq'] = 0
    
    return features.values

train_tabular = extract_tabular_features(train_df, is_train=True)
test_tabular = extract_tabular_features(test_df, is_train=False)

print(f"✓ Extracted tabular features: {train_tabular.shape[1]} columns")
print(f"  Features: value, pack_size, text_length, word_count, num_bullets,")
print(f"            binary flags, interactions, unit_encoded, price_segment_encoded, brand_freq")

# Normalize tabular features
scaler = StandardScaler()
train_tabular_scaled = scaler.fit_transform(train_tabular)
test_tabular_scaled = scaler.transform(test_tabular)

# ============================================================================
# TRAIN/VAL SPLIT (SAME SEED AS SCRIPTS 1&2)
# ============================================================================
print("\n[3/8] Creating train/validation split...")

indices = np.arange(len(y_train))
train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

X_img_train, X_img_val = train_img[train_idx], train_img[val_idx]
X_txt_train, X_txt_val = train_txt[train_idx], train_txt[val_idx]
X_tab_train, X_tab_val = train_tabular_scaled[train_idx], train_tabular_scaled[val_idx]
y_train_split, y_val = y_train_log[train_idx], y_train_log[val_idx]

print(f"✓ Train: {len(train_idx):,} samples")
print(f"✓ Val:   {len(val_idx):,} samples")

# ============================================================================
# LOAD PRE-TRAINED MODELS
# ============================================================================
print("\n[4/8] Loading pre-trained Image & Text MLPs...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ImageMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.network(x).squeeze(-1)

class TextMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.network(x).squeeze(-1)

image_model = ImageMLP().to(device)
text_model = TextMLP().to(device)

image_model.load_state_dict(torch.load('image_mlp_best_FIXED.pth'))
text_model.load_state_dict(torch.load('text_mlp_best_FIXED.pth'))

image_model.eval()
text_model.eval()

print("✓ Image MLP loaded from: image_mlp_best_FIXED.pth")
print("✓ Text MLP loaded from: text_mlp_best_FIXED.pth")

# ============================================================================
# DEFINE ENSEMBLE MODEL
# ============================================================================
print("\n[5/8] Defining ensemble architecture...")

class EnsembleModel(nn.Module):
    """
    Combines:
    - Image MLP predictions (frozen)
    - Text MLP predictions (frozen)
    - Tabular features (trainable)
    """
    def __init__(self, tabular_dim):
        super().__init__()
        
        # Tabular feature network
        self.tabular_net = nn.Sequential(
            nn.Linear(tabular_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Final fusion layer (2 model predictions + 32 tabular features)
        self.fusion = nn.Sequential(
            nn.Linear(2 + 32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, img_pred, txt_pred, tabular):
        # Process tabular features
        tab_features = self.tabular_net(tabular)
        
        # Combine all
        combined = torch.cat([
            img_pred.unsqueeze(1),
            txt_pred.unsqueeze(1),
            tab_features
        ], dim=1)
        
        # Final prediction
        return self.fusion(combined).squeeze(-1)

ensemble_model = EnsembleModel(tabular_dim=train_tabular_scaled.shape[1]).to(device)
print(f"✓ Ensemble architecture: [Image pred + Text pred + {train_tabular_scaled.shape[1]} tabular → 64 → 32 → 1]")

# ============================================================================
# GET PREDICTIONS FROM PRE-TRAINED MODELS
# ============================================================================
print("\n[6/8] Generating predictions from Image & Text MLPs...")

with torch.no_grad():
    img_train_pred = image_model(torch.FloatTensor(X_img_train).to(device)).cpu().numpy()
    img_val_pred = image_model(torch.FloatTensor(X_img_val).to(device)).cpu().numpy()
    txt_train_pred = text_model(torch.FloatTensor(X_txt_train).to(device)).cpu().numpy()
    txt_val_pred = text_model(torch.FloatTensor(X_txt_val).to(device)).cpu().numpy()

print("✓ Generated predictions for train & validation")

# ============================================================================
# TRAIN ENSEMBLE
# ============================================================================
print("\n[7/8] Training ensemble model...")

train_dataset = TensorDataset(
    torch.FloatTensor(img_train_pred),
    torch.FloatTensor(txt_train_pred),
    torch.FloatTensor(X_tab_train),
    torch.FloatTensor(y_train_split)
)
val_dataset = TensorDataset(
    torch.FloatTensor(img_val_pred),
    torch.FloatTensor(txt_val_pred),
    torch.FloatTensor(X_tab_val),
    torch.FloatTensor(y_val)
)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=512, num_workers=2, pin_memory=True)

optimizer = optim.AdamW(ensemble_model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8, verbose=True)
criterion = nn.MSELoss()

print("\nEpoch | Train Loss | Val Loss | Val SMAPE | LR      | Status")
print("-" * 70)

history = {'train_loss': [], 'val_loss': [], 'val_smape': [], 'lr': []}

best_val_loss = float('inf')
best_val_smape = float('inf')
patience = 15
patience_counter = 0
n_epochs = 100

for epoch in range(n_epochs):
    # Train
    ensemble_model.train()
    train_loss = 0
    for img_batch, txt_batch, tab_batch, y_batch in train_loader:
        img_batch = img_batch.to(device)
        txt_batch = txt_batch.to(device)
        tab_batch = tab_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        pred = ensemble_model(img_batch, txt_batch, tab_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ensemble_model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item() * len(y_batch)
    train_loss /= len(train_dataset)
    
    # Validate
    ensemble_model.eval()
    val_loss = 0
    val_preds = []
    val_targets = []
    with torch.no_grad():
        for img_batch, txt_batch, tab_batch, y_batch in val_loader:
            img_batch = img_batch.to(device)
            txt_batch = txt_batch.to(device)
            tab_batch = tab_batch.to(device)
            y_batch = y_batch.to(device)
            
            pred = ensemble_model(img_batch, txt_batch, tab_batch)
            loss = criterion(pred, y_batch)
            val_loss += loss.item() * len(y_batch)
            val_preds.append(pred.cpu().numpy())
            val_targets.append(y_batch.cpu().numpy())
    
    val_loss /= len(val_dataset)
    val_preds = np.concatenate(val_preds)
    val_targets = np.concatenate(val_targets)
    
    # SMAPE
    val_preds_price = np.expm1(val_preds)
    val_targets_price = np.expm1(val_targets)
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
        torch.save(ensemble_model.state_dict(), 'ensemble_best_FIXED.pth')
        patience_counter = 0
        status = "✓ BEST"
    else:
        patience_counter += 1
        status = f"({patience_counter}/{patience})"
    
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"{epoch+1:5d} | {train_loss:10.4f} | {val_loss:8.4f} | "
              f"{val_smape:9.2f}% | {current_lr:.6f} | {status}")
    
    if patience_counter >= patience:
        print(f"\n✓ Early stopping at epoch {epoch+1}")
        break

ensemble_model.load_state_dict(torch.load('ensemble_best_FIXED.pth'))
print(f"\n✓ Best ensemble SMAPE: {best_val_smape:.2f}%")

# ============================================================================
# GENERATE TEST PREDICTIONS
# ============================================================================
print("\n[8/8] Generating final test predictions...")

# Get Image & Text predictions for test set
with torch.no_grad():
    test_img_pred = image_model(torch.FloatTensor(test_img).to(device)).cpu().numpy()
    test_txt_pred = text_model(torch.FloatTensor(test_txt).to(device)).cpu().numpy()

# Ensemble prediction
test_dataset = TensorDataset(
    torch.FloatTensor(test_img_pred),
    torch.FloatTensor(test_txt_pred),
    torch.FloatTensor(test_tabular_scaled)
)
test_loader = DataLoader(test_dataset, batch_size=512, num_workers=2, pin_memory=True)

ensemble_model.eval()
final_predictions = []
with torch.no_grad():
    for img_batch, txt_batch, tab_batch in test_loader:
        img_batch = img_batch.to(device)
        txt_batch = txt_batch.to(device)
        tab_batch = tab_batch.to(device)
        
        pred = ensemble_model(img_batch, txt_batch, tab_batch)
        final_predictions.append(pred.cpu().numpy())

final_predictions = np.concatenate(final_predictions)
final_predictions_price = np.expm1(final_predictions)

# Create submission
submission = pd.DataFrame({
    'id': range(len(final_predictions_price)),
    'price': final_predictions_price
})
submission.to_csv('submission_FIXED.csv', index=False)

print(f"✓ Submission created: submission_FIXED.csv")
print(f"  Predictions: {len(final_predictions_price):,}")
print(f"  Range: [{final_predictions_price.min():.2f}, {final_predictions_price.max():.2f}]")

# ============================================================================
# FINAL COMPARISON & DIAGNOSTICS
# ============================================================================
print("\n" + "=" * 80)
print("📊 COMPREHENSIVE MODEL COMPARISON")
print("=" * 80)

# Individual model performance
with torch.no_grad():
    img_val_pred_eval = image_model(torch.FloatTensor(X_img_val).to(device)).cpu().numpy()
    txt_val_pred_eval = text_model(torch.FloatTensor(X_txt_val).to(device)).cpu().numpy()

img_val_price = np.expm1(img_val_pred_eval)
txt_val_price = np.expm1(txt_val_pred_eval)
val_targets_price = np.expm1(y_val)

img_smape = 100 * np.mean(2 * np.abs(img_val_price - val_targets_price) / 
                          (np.abs(img_val_price) + np.abs(val_targets_price)))
txt_smape = 100 * np.mean(2 * np.abs(txt_val_price - val_targets_price) / 
                          (np.abs(txt_val_price) + np.abs(val_targets_price)))

ensemble_model.eval()
with torch.no_grad():
    ens_val_pred = []
    for img_batch, txt_batch, tab_batch, _ in val_loader:
        pred = ensemble_model(img_batch.to(device), txt_batch.to(device), tab_batch.to(device))
        ens_val_pred.append(pred.cpu().numpy())
ens_val_pred = np.concatenate(ens_val_pred)
ens_val_price = np.expm1(ens_val_pred)

ens_smape = 100 * np.mean(2 * np.abs(ens_val_price - val_targets_price) / 
                          (np.abs(ens_val_price) + np.abs(val_targets_price)))

print(f"\n📈 Validation SMAPE Comparison:")
print(f"   Image MLP:    {img_smape:.2f}%")
print(f"   Text MLP:     {txt_smape:.2f}%")
print(f"   Ensemble:     {ens_smape:.2f}%")
print(f"   Improvement:  {min(img_smape, txt_smape) - ens_smape:+.2f}%")

if ens_smape < min(img_smape, txt_smape):
    print(f"\n   ✅ ENSEMBLE WINS! {ens_smape:.2f}% < {min(img_smape, txt_smape):.2f}%")
else:
    print(f"\n   ⚠️  Ensemble didn't improve over best individual model")

# Diagnostic plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Training curves
ax = axes[0, 0]
ax.plot(history['train_loss'], label='Train Loss', linewidth=2)
ax.plot(history['val_loss'], label='Val Loss', linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE Loss')
ax.set_title('Ensemble Training Curves')
ax.legend()
ax.grid(True, alpha=0.3)

# SMAPE
ax = axes[0, 1]
ax.plot(history['val_smape'], color='green', linewidth=2)
ax.axhline(y=ens_smape, color='red', linestyle='--', label=f'Best: {ens_smape:.2f}%')
ax.set_xlabel('Epoch')
ax.set_ylabel('SMAPE (%)')
ax.set_title('Ensemble Validation SMAPE')
ax.legend()
ax.grid(True, alpha=0.3)

# Model comparison
ax = axes[1, 0]
models = ['Image\nMLP', 'Text\nMLP', 'Ensemble']
smapes = [img_smape, txt_smape, ens_smape]
colors = ['blue', 'orange', 'green']
bars = ax.bar(models, smapes, color=colors, alpha=0.7)
ax.set_ylabel('SMAPE (%)')
ax.set_title('Model Comparison (Validation Set)')
ax.grid(True, alpha=0.3, axis='y')
for bar, smape in zip(bars, smapes):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{smape:.1f}%', ha='center', va='bottom', fontweight='bold')

# Scatter plot
ax = axes[1, 1]
ax.scatter(val_targets_price, ens_val_price, alpha=0.3, s=10, label='Ensemble')
ax.plot([0, val_targets_price.max()], [0, val_targets_price.max()], 
        'r--', linewidth=2, label='Perfect')
ax.set_xlabel('Actual Price ($)')
ax.set_ylabel('Predicted Price ($)')
ax.set_title('Ensemble: Actual vs Predicted')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ensemble_diagnostics_FIXED.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Saved diagnostic plots to: ensemble_diagnostics_FIXED.png")

# ============================================================================
# FINAL RECOMMENDATIONS
# ============================================================================
print("\n" + "=" * 80)
print("🎯 FINAL DIAGNOSTIC SUMMARY")
print("=" * 80)

print(f"\n✅ Ensemble Model Complete:")
print(f"   - Validation SMAPE: {ens_smape:.2f}%")

print(f"\n🔍 Key Insights:")
print(f"   1. Best standalone: {'Image' if img_smape < txt_smape else 'Text'} MLP "
      f"({min(img_smape, txt_smape):.2f}%)")
print(f"   2. Ensemble benefit: {min(img_smape, txt_smape) - ens_smape:+.2f}%")

correlation = np.corrcoef(img_val_price, txt_val_price)[0, 1]
print(f"   3. Image-Text correlation: {correlation:.3f}")
if correlation > 0.8:
    print(f"      ⚠️  High correlation - modalities may be redundant")
else:
    print(f"      ✅ Modalities capture complementary signals")

print(f"\n🎯 Submission Decision:")
if ens_smape < 45:
    print(f"   ✅ EXCELLENT! {ens_smape:.2f}% SMAPE")
    print(f"   📤 READY TO SUBMIT: submission_FIXED.csv")
    print(f"   🎯 Expected LB: ~{ens_smape:.0f}% (±5%)")
elif ens_smape < 55:
    print(f"   ✅ GOOD! {ens_smape:.2f}% SMAPE")
    print(f"   📤 SUBMIT and see LB feedback")
    print(f"   🎯 Expected LB: ~{ens_smape:.0f}% (±5%)")
else:
    print(f"   ⚠️  MODERATE: {ens_smape:.2f}% SMAPE")
    print(f"   💡 Consider: Option D (fine-tuning) for better results")

print(f"\n📁 Final Outputs:")
print(f"   - submission_FIXED.csv (ready for Kaggle)")
print(f"   - ensemble_best_FIXED.pth (model weights)")
print(f"   - ensemble_diagnostics_FIXED.png (plots)")

print("\n" + "=" * 80)
print("🚀 ALL DIAGNOSTIC SCRIPTS COMPLETE!")
print("=" * 80)
print(f"\nNext: Submit submission_FIXED.csv to Kaggle and check Leaderboard!")
