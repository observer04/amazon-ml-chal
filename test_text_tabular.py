"""
DIAGNOSTIC: Text + Tabular Ensemble (Skip Images)
===================================================
Purpose: Test if text embeddings + tabular features can achieve target SMAPE
Expected: 40-50% SMAPE (vs 73% with broken image embeddings)

Why skip images:
- Image R² = -0.0085 (worse than baseline)
- Text R² = 0.3345 (useful!)
- Tabular features likely strong (value, pack_size, etc.)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("DIAGNOSTIC: Text + Tabular Ensemble (NO IMAGES)")
print("Testing if we can hit 40-50% SMAPE without broken image embeddings")
print("=" * 80)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n[1/6] Loading text embeddings and features...")

txt_emb = np.load('outputs/train_text_embeddings_clip.npy')
train_df = pd.read_csv('dataset/train_with_features.csv', low_memory=False)
y_train = train_df['price'].values
y_train_log = np.log1p(y_train)

print(f"✓ Text embeddings: {txt_emb.shape}")
print(f"✓ Price range: [${y_train.min():.2f}, ${y_train.max():.2f}]")

# ============================================================================
# EXTRACT TABULAR FEATURES
# ============================================================================
print("\n[2/6] Extracting tabular features...")

def extract_tabular_features(df):
    features = pd.DataFrame()
    
    # Numerical features
    numerical_cols = ['value', 'pack_size', 'text_length', 'word_count', 
                     'num_bullets', 'description_length', 'price_per_unit']
    for col in numerical_cols:
        if col in df.columns:
            features[col] = df[col].fillna(0)
    
    # Binary flags
    binary_cols = ['has_premium', 'has_organic', 'has_gourmet', 'has_natural', 
                  'has_artisan', 'has_luxury', 'is_travel_size', 'is_bulk', 
                  'brand_exists', 'has_bullets']
    for col in binary_cols:
        if col in df.columns:
            features[col] = df[col].fillna(0).astype(int)
    
    # Interaction features
    interaction_cols = ['value_x_premium', 'value_x_luxury', 'value_x_organic',
                       'pack_x_premium', 'pack_x_value', 'brand_x_premium',
                       'brand_x_organic', 'travel_x_value', 'bulk_x_value']
    for col in interaction_cols:
        if col in df.columns:
            features[col] = df[col].fillna(0)
    
    # Categorical encoding
    if 'unit' in df.columns:
        unit_map = {'kilogram': 0, 'gram': 1, 'litre': 2, 'millilitre': 3, 
                   'centimetre': 4, 'metre': 5, 'unit': 6, 'count': 7}
        features['unit_encoded'] = df['unit'].fillna('unit').map(unit_map).fillna(6)
    
    if 'price_segment' in df.columns:
        segment_map = {'budget': 0, 'economy': 1, 'mid_range': 2, 'premium': 3, 'luxury': 4}
        features['price_segment_encoded'] = df['price_segment'].fillna('mid_range').map(segment_map).fillna(2)
    
    # Brand frequency
    if 'brand' in df.columns:
        brand_counts = df['brand'].value_counts()
        features['brand_freq'] = df['brand'].map(brand_counts).fillna(0) / len(df)
    
    return features.values

tabular = extract_tabular_features(train_df)
print(f"✓ Extracted {tabular.shape[1]} tabular features")

# Normalize
scaler = StandardScaler()
tabular_scaled = scaler.fit_transform(tabular)

# ============================================================================
# TEST TABULAR FEATURES ALONE
# ============================================================================
print("\n[3/6] Testing tabular features predictive power...")

from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

indices = np.arange(len(y_train))
train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

X_tab_train, X_tab_val = tabular_scaled[train_idx], tabular_scaled[val_idx]
y_log_train, y_log_val = y_train_log[train_idx], y_train_log[val_idx]

model_tab = Ridge(alpha=1.0)
model_tab.fit(X_tab_train, y_log_train)
pred_tab = model_tab.predict(X_tab_val)

r2_tab = r2_score(y_log_val, pred_tab)
pred_tab_price = np.expm1(pred_tab)
actual_val_price = np.expm1(y_log_val)
smape_tab = 100 * np.mean(2 * np.abs(pred_tab_price - actual_val_price) / 
                          (np.abs(pred_tab_price) + np.abs(actual_val_price)))

print(f"Tabular features alone:")
print(f"  R²: {r2_tab:.4f}")
print(f"  SMAPE: {smape_tab:.2f}%")

# Test text
X_txt_train, X_txt_val = txt_emb[train_idx], txt_emb[val_idx]

model_txt = Ridge(alpha=1.0)
model_txt.fit(X_txt_train, y_log_train)
pred_txt = model_txt.predict(X_txt_val)

r2_txt = r2_score(y_log_val, pred_txt)
pred_txt_price = np.expm1(pred_txt)
smape_txt = 100 * np.mean(2 * np.abs(pred_txt_price - actual_val_price) / 
                          (np.abs(pred_txt_price) + np.abs(actual_val_price)))

print(f"\nText embeddings alone:")
print(f"  R²: {r2_txt:.4f}")
print(f"  SMAPE: {smape_txt:.2f}%")

# Test combination
X_combined_train = np.concatenate([X_txt_train, X_tab_train], axis=1)
X_combined_val = np.concatenate([X_txt_val, X_tab_val], axis=1)

model_combined = Ridge(alpha=1.0)
model_combined.fit(X_combined_train, y_log_train)
pred_combined = model_combined.predict(X_combined_val)

r2_combined = r2_score(y_log_val, pred_combined)
pred_combined_price = np.expm1(pred_combined)
smape_combined = 100 * np.mean(2 * np.abs(pred_combined_price - actual_val_price) / 
                               (np.abs(pred_combined_price) + np.abs(actual_val_price)))

print(f"\nText + Tabular combined:")
print(f"  R²: {r2_combined:.4f}")
print(f"  SMAPE: {smape_combined:.2f}%")

print(f"\n📊 SUMMARY:")
print(f"  Tabular only:  {smape_tab:.2f}% SMAPE")
print(f"  Text only:     {smape_txt:.2f}% SMAPE")
print(f"  Combined:      {smape_combined:.2f}% SMAPE")
print(f"  Improvement:   {min(smape_tab, smape_txt) - smape_combined:+.2f}%")

# ============================================================================
# TRAIN MLP ON TEXT + TABULAR
# ============================================================================
print("\n[4/6] Training MLP on Text + Tabular features...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class TextTabularMLP(nn.Module):
    def __init__(self, text_dim=512, tabular_dim=30):
        super().__init__()
        
        # Text processing branch
        self.text_net = nn.Sequential(
            nn.Linear(text_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Tabular processing branch
        self.tabular_net = nn.Sequential(
            nn.Linear(tabular_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(128 + 32, 128),
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
    
    def forward(self, text, tabular):
        text_features = self.text_net(text)
        tabular_features = self.tabular_net(tabular)
        combined = torch.cat([text_features, tabular_features], dim=1)
        return self.fusion(combined).squeeze(-1)

model = TextTabularMLP(text_dim=512, tabular_dim=tabular_scaled.shape[1]).to(device)

train_dataset = TensorDataset(
    torch.FloatTensor(X_txt_train),
    torch.FloatTensor(X_tab_train),
    torch.FloatTensor(y_log_train)
)
val_dataset = TensorDataset(
    torch.FloatTensor(X_txt_val),
    torch.FloatTensor(X_tab_val),
    torch.FloatTensor(y_log_val)
)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0, pin_memory=False)
val_loader = DataLoader(val_dataset, batch_size=512, num_workers=0, pin_memory=False)

optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=False)
criterion = nn.MSELoss()

print("\n[5/6] Training...")
print("Epoch | Train Loss | Val Loss | Val SMAPE | Status")
print("-" * 60)

best_val_smape = float('inf')
patience = 20
patience_counter = 0
n_epochs = 100

for epoch in range(n_epochs):
    # Train
    model.train()
    train_loss = 0
    for txt_batch, tab_batch, y_batch in train_loader:
        txt_batch = txt_batch.to(device)
        tab_batch = tab_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        pred = model(txt_batch, tab_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item() * len(y_batch)
    train_loss /= len(train_dataset)
    
    # Validate
    model.eval()
    val_loss = 0
    val_preds = []
    with torch.no_grad():
        for txt_batch, tab_batch, y_batch in val_loader:
            txt_batch = txt_batch.to(device)
            tab_batch = tab_batch.to(device)
            y_batch = y_batch.to(device)
            
            pred = model(txt_batch, tab_batch)
            loss = criterion(pred, y_batch)
            val_loss += loss.item() * len(y_batch)
            val_preds.append(pred.cpu().numpy())
    
    val_loss /= len(val_dataset)
    val_preds = np.concatenate(val_preds)
    
    # SMAPE
    val_preds_price = np.expm1(val_preds)
    val_smape = 100 * np.mean(2 * np.abs(val_preds_price - actual_val_price) / 
                              (np.abs(val_preds_price) + np.abs(actual_val_price)))
    
    scheduler.step(val_loss)
    
    status = ""
    if val_smape < best_val_smape:
        best_val_smape = val_smape
        torch.save(model.state_dict(), 'text_tabular_best.pth')
        patience_counter = 0
        status = "✓ BEST"
    else:
        patience_counter += 1
        status = f"({patience_counter}/{patience})"
    
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"{epoch+1:5d} | {train_loss:10.4f} | {val_loss:8.4f} | {val_smape:9.2f}% | {status}")
    
    if patience_counter >= patience:
        print(f"\n✓ Early stopping at epoch {epoch+1}")
        break

model.load_state_dict(torch.load('text_tabular_best.pth'))

# ============================================================================
# FINAL EVALUATION
# ============================================================================
print("\n[6/6] Final evaluation...")

model.eval()
with torch.no_grad():
    final_preds = []
    for txt_batch, tab_batch, _ in val_loader:
        txt_batch = txt_batch.to(device)
        tab_batch = tab_batch.to(device)
        pred = model(txt_batch, tab_batch)
        final_preds.append(pred.cpu().numpy())

final_preds = np.concatenate(final_preds)
final_preds_price = np.expm1(final_preds)

final_smape = 100 * np.mean(2 * np.abs(final_preds_price - actual_val_price) / 
                            (np.abs(final_preds_price) + np.abs(actual_val_price)))

# By price range
print(f"\n📈 Performance by Price Range:")
for low, high in [(0, 10), (10, 25), (25, 50), (50, 100), (100, 1000), (1000, 10000)]:
    mask = (actual_val_price >= low) & (actual_val_price < high)
    if mask.sum() > 0:
        range_smape = 100 * np.mean(
            2 * np.abs(final_preds_price[mask] - actual_val_price[mask]) / 
            (np.abs(final_preds_price[mask]) + np.abs(actual_val_price[mask]))
        )
        print(f"   ${low:4d}-${high:5d}: {range_smape:6.2f}% SMAPE ({mask.sum():5d} samples)")

print("\n" + "=" * 80)
print("📊 FINAL RESULTS")
print("=" * 80)

print(f"\n🎯 Text + Tabular MLP:")
print(f"   Best Validation SMAPE: {best_val_smape:.2f}%")
print(f"   Prediction range: [${final_preds_price.min():.2f}, ${final_preds_price.max():.2f}]")
print(f"   Actual range: [${actual_val_price.min():.2f}, ${actual_val_price.max():.2f}]")

print(f"\n📊 Comparison with Image MLP:")
print(f"   Image MLP (BROKEN):    73.85% SMAPE")
print(f"   Text + Tabular:        {best_val_smape:.2f}% SMAPE")
print(f"   Improvement:           {73.85 - best_val_smape:+.2f}%")

print(f"\n💡 Analysis:")
if best_val_smape < 45:
    print(f"   ✅ EXCELLENT! {best_val_smape:.2f}% meets target (<45%)")
    print(f"   📤 Ready to create submission!")
elif best_val_smape < 55:
    print(f"   ✅ GOOD! {best_val_smape:.2f}% much better than image-based")
    print(f"   💡 Consider deeper network or more features")
else:
    print(f"   ⚠️  MODERATE: {best_val_smape:.2f}% - still better than images")
    print(f"   💡 May need feature engineering or different approach")

print(f"\n🎯 Next Steps:")
if best_val_smape < 55:
    print(f"   1. Create full training script for Kaggle")
    print(f"   2. Generate test predictions")
    print(f"   3. Submit to leaderboard")
else:
    print(f"   1. Analyze which features are most important")
    print(f"   2. Engineer better features")
    print(f"   3. Try ensemble methods")

print("=" * 80)
