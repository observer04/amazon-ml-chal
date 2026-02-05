"""
DIAGNOSTIC SCRIPT 3: FINAL ENSEMBLE & SUBMISSION
=================================================
This script:
1. Loads best Image MLP and Text MLP from diagnostics
2. Trains Tabular LightGBM
3. Optimizes ensemble weights based on validation performance
4. Generates final submission

IMPORTANT: Run Scripts 1 & 2 FIRST!
- KAGGLE_script1_image_mlp.py
- KAGGLE_script2_text_mlp.py

Time: 30 minutes
GPU: Required
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("DIAGNOSTIC 3: FINAL ENSEMBLE & SUBMISSION")
print("Combining: Image MLP + Text MLP + Tabular LightGBM")
print("=" * 80)

# ============================================================================
# LOAD ALL DATA
# ============================================================================
print("\n[1/6] Loading all embeddings and features...")

train_img_emb = np.load('/kaggle/working/amazon-ml-chal/outputs/train_image_embeddings_clip.npy')
train_txt_emb = np.load('/kaggle/working/amazon-ml-chal/outputs/train_text_embeddings_clip.npy')
test_img_emb = np.load('/kaggle/working/amazon-ml-chal/outputs/test_image_embeddings_clip.npy')
test_txt_emb = np.load('/kaggle/working/amazon-ml-chal/outputs/test_text_embeddings_clip.npy')

train_df = pd.read_csv('/kaggle/working/amazon-ml-chal/dataset/train_with_features.csv')
test_df = pd.read_csv('/kaggle/working/amazon-ml-chal/dataset/test_with_features.csv')

# Feature columns (30 engineered features from baseline)
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
y_train_sqrt = np.sqrt(y_train)

print(f"✓ Image embeddings: Train {train_img_emb.shape}, Test {test_img_emb.shape}")
print(f"✓ Text embeddings:  Train {train_txt_emb.shape}, Test {test_txt_emb.shape}")
print(f"✓ Tabular features: Train {X_tab_train.shape}, Test {X_tab_test.shape}")
print(f"✓ All data loaded")

# Same train/val split as diagnostics (CRITICAL: use same random_state=42)
indices = np.arange(len(y_train))
train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

# ============================================================================
# LOAD TRAINED MLP MODELS
# ============================================================================
print("\n[2/6] Loading trained MLP models from diagnostics...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✓ Device: {device}")

# Define model architecture (must match diagnostic scripts)
class MLP(nn.Module):
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
image_mlp = MLP().to(device)
image_mlp.load_state_dict(torch.load('image_mlp_best.pth'))
image_mlp.eval()
print("✓ Image MLP loaded")

# Load Text MLP
text_mlp = MLP().to(device)
text_mlp.load_state_dict(torch.load('text_mlp_best.pth'))
text_mlp.eval()
print("✓ Text MLP loaded")

# Get validation predictions from MLPs
X_img_val = train_img_emb[val_idx]
X_txt_val = train_txt_emb[val_idx]
y_val = y_train_sqrt[val_idx]

with torch.no_grad():
    val_pred_img = image_mlp(torch.FloatTensor(X_img_val).to(device)).cpu().numpy()
    val_pred_txt = text_mlp(torch.FloatTensor(X_txt_val).to(device)).cpu().numpy()

val_pred_img_price = val_pred_img ** 2
val_pred_txt_price = val_pred_txt ** 2
y_val_price = y_val ** 2

# Calculate individual SMAPEs
smape_img = 100 * np.mean(2 * np.abs(val_pred_img_price - y_val_price) / (np.abs(val_pred_img_price) + np.abs(y_val_price)))
smape_txt = 100 * np.mean(2 * np.abs(val_pred_txt_price - y_val_price) / (np.abs(val_pred_txt_price) + np.abs(y_val_price)))

print(f"   Image MLP SMAPE: {smape_img:.2f}%")
print(f"   Text MLP SMAPE:  {smape_txt:.2f}%")

# ============================================================================
# TRAIN TABULAR LIGHTGBM
# ============================================================================
print("\n[3/6] Training Tabular LightGBM...")

X_tab_train_split = X_tab_train[train_idx]
X_tab_val = X_tab_train[val_idx]
y_train_split = y_train_sqrt[train_idx]

lgb_train = lgb.Dataset(X_tab_train_split, y_train_split)
lgb_val = lgb.Dataset(X_tab_val, y_val, reference=lgb_train)

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

print("Training...")
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

print(f"✓ Tabular LightGBM trained (best iteration: {lgb_model.best_iteration})")

val_pred_tab = lgb_model.predict(X_tab_val, num_iteration=lgb_model.best_iteration)
val_pred_tab_price = val_pred_tab ** 2
smape_tab = 100 * np.mean(2 * np.abs(val_pred_tab_price - y_val_price) / (np.abs(val_pred_tab_price) + np.abs(y_val_price)))

print(f"   Tabular LGBM SMAPE: {smape_tab:.2f}%")

# ============================================================================
# OPTIMIZE ENSEMBLE WEIGHTS
# ============================================================================
print("\n[4/6] Optimizing ensemble weights...")

print(f"\n📊 Individual Branch Performance:")
print(f"   Branch 1 (Tabular LGBM): {smape_tab:.2f}% SMAPE")
print(f"   Branch 2 (Text MLP):     {smape_txt:.2f}% SMAPE")
print(f"   Branch 3 (Image MLP):    {smape_img:.2f}% SMAPE")
print(f"   Best single:             {min(smape_tab, smape_txt, smape_img):.2f}% SMAPE")

def ensemble_smape(weights, pred_tab, pred_txt, pred_img, y_true):
    """Objective function for weight optimization"""
    w_tab, w_txt, w_img = weights
    ensemble_pred = w_tab * pred_tab + w_txt * pred_txt + w_img * pred_img
    smape = 100 * np.mean(2 * np.abs(ensemble_pred - y_true) / (np.abs(ensemble_pred) + np.abs(y_true)))
    return smape

def objective(weights):
    return ensemble_smape(weights, val_pred_tab_price, val_pred_txt_price, val_pred_img_price, y_val_price)

# Constraints: weights sum to 1, all non-negative
constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
bounds = [(0, 1), (0, 1), (0, 1)]
initial_weights = [0.2, 0.3, 0.5]  # Educated guess: image > text > tabular

print(f"\nStarting optimization...")
print(f"   Initial weights [tab, txt, img]: {initial_weights}")
print(f"   Initial ensemble SMAPE: {objective(initial_weights):.2f}%")

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
print(f"   Optimal weights:")
print(f"      Tabular: {optimal_weights[0]:.3f}")
print(f"      Text:    {optimal_weights[1]:.3f}")
print(f"      Image:   {optimal_weights[2]:.3f}")
print(f"   Optimal SMAPE: {ensemble_smape_final:.2f}%")

improvement = min(smape_tab, smape_txt, smape_img) - ensemble_smape_final
print(f"\n📈 Improvement over best single model: {improvement:.2f}% SMAPE")

if improvement > 0:
    print("   ✅ Ensemble is better than individual models!")
else:
    print("   ⚠️  Ensemble not better - using best single model")
    best_idx = np.argmin([smape_tab, smape_txt, smape_img])
    optimal_weights = [[1, 0, 0], [0, 1, 0], [0, 0, 1]][best_idx]
    print(f"   Fallback weights: {optimal_weights}")

# ============================================================================
# GENERATE TEST PREDICTIONS
# ============================================================================
print("\n[5/6] Generating final test predictions...")

print("Re-training all models on full training data...")

# --- Image MLP on full train ---
print("  [1/3] Image MLP...")
train_dataset_img_full = TensorDataset(
    torch.FloatTensor(train_img_emb),
    torch.FloatTensor(y_train_sqrt)
)
train_loader_img_full = DataLoader(train_dataset_img_full, batch_size=256, shuffle=True, num_workers=2)

image_mlp_final = MLP().to(device)
optimizer_img_final = torch.optim.AdamW(image_mlp_final.parameters(), lr=0.001, weight_decay=1e-5)
criterion = nn.MSELoss()

for epoch in range(50):  # Fewer epochs, no validation
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

text_mlp_final = MLP().to(device)
optimizer_txt_final = torch.optim.AdamW(text_mlp_final.parameters(), lr=0.001, weight_decay=1e-5)

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
    callbacks=[lgb.log_evaluation(0)]
)
print("        ✓ LightGBM trained on 75k samples")

# Generate test predictions
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
test_pred_ensemble = np.clip(test_pred_ensemble, 0.01, None)

print(f"✓ Ensemble predictions range: [{test_pred_ensemble.min():.2f}, {test_pred_ensemble.max():.2f}]")

# ============================================================================
# SAVE SUBMISSION
# ============================================================================
print("\n[6/6] Creating submission file...")

submission = pd.DataFrame({
    'sample_id': test_df['sample_id'],
    'price': test_pred_ensemble
})

submission.to_csv('submission_multimodal_ensemble.csv', index=False)

print(f"✓ Submission saved: submission_multimodal_ensemble.csv")
print(f"   Shape: {submission.shape}")
print(f"\n📊 Sample predictions:")
print(submission.head(10))

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("\nGenerating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Model comparison
ax = axes[0, 0]
models = ['Tabular', 'Text', 'Image', 'Ensemble']
smapes = [smape_tab, smape_txt, smape_img, ensemble_smape_final]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
bars = ax.bar(models, smapes, color=colors, alpha=0.8, edgecolor='black')
ax.set_ylabel('SMAPE (%)', fontsize=12, fontweight='bold')
ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
for bar, smape in zip(bars, smapes):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{smape:.1f}%', ha='center', va='bottom', fontweight='bold')

# Plot 2: Ensemble weights
ax = axes[0, 1]
weights_labels = ['Tabular', 'Text', 'Image']
colors_pie = ['#FF6B6B', '#4ECDC4', '#45B7D1']
ax.pie(optimal_weights, labels=weights_labels, autopct='%1.1f%%',
       colors=colors_pie, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
ax.set_title('Optimal Ensemble Weights', fontsize=14, fontweight='bold')

# Plot 3: Prediction distribution
ax = axes[1, 0]
ax.hist(test_pred_ensemble, bins=50, alpha=0.7, color='#96CEB4', edgecolor='black')
ax.axvline(test_pred_ensemble.mean(), color='red', linestyle='--', linewidth=2, 
           label=f'Mean: ${test_pred_ensemble.mean():.2f}')
ax.axvline(np.median(test_pred_ensemble), color='blue', linestyle='--', linewidth=2, 
           label=f'Median: ${np.median(test_pred_ensemble):.2f}')
ax.set_xlabel('Predicted Price ($)', fontsize=12, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax.set_title('Test Set Prediction Distribution', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Plot 4: Individual model predictions comparison
ax = axes[1, 1]
sample_size = min(1000, len(test_pred_tab_price))
sample_idx = np.random.choice(len(test_pred_tab_price), sample_size, replace=False)
ax.scatter(test_pred_tab_price[sample_idx], test_pred_img_price[sample_idx], 
           alpha=0.5, s=10, label='Tab vs Img')
ax.scatter(test_pred_tab_price[sample_idx], test_pred_txt_price[sample_idx], 
           alpha=0.5, s=10, label='Tab vs Txt')
ax.plot([test_pred_tab_price[sample_idx].min(), test_pred_tab_price[sample_idx].max()],
        [test_pred_tab_price[sample_idx].min(), test_pred_tab_price[sample_idx].max()],
        'r--', linewidth=2, label='Perfect Agreement')
ax.set_xlabel('Tabular Predictions ($)', fontsize=12, fontweight='bold')
ax.set_ylabel('Other Model Predictions ($)', fontsize=12, fontweight='bold')
ax.set_title('Model Agreement Analysis', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('ensemble_analysis.png', dpi=150, bbox_inches='tight')
print("✓ Visualizations saved: ensemble_analysis.png")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("✅ ENSEMBLE TRAINING COMPLETE!")
print("=" * 80)

print(f"\n📊 Final Results:")
print(f"   Individual Models:")
print(f"      - Tabular LightGBM: {smape_tab:.2f}% SMAPE")
print(f"      - Text MLP:         {smape_txt:.2f}% SMAPE")
print(f"      - Image MLP:        {smape_img:.2f}% SMAPE")
print(f"   Ensemble:              {ensemble_smape_final:.2f}% SMAPE")
print(f"   Improvement:           {improvement:+.2f}%")

print(f"\n🎯 Optimal Weights:")
print(f"   Tabular: {optimal_weights[0]:.1%}")
print(f"   Text:    {optimal_weights[1]:.1%}")
print(f"   Image:   {optimal_weights[2]:.1%}")

print(f"\n📁 Generated Files:")
print(f"   - submission_multimodal_ensemble.csv")
print(f"   - ensemble_analysis.png")

print(f"\n🚀 Next Steps:")
print(f"   1. Review ensemble_analysis.png for insights")
print(f"   2. Submit submission_multimodal_ensemble.csv to Kaggle")
print(f"   3. Compare leaderboard score with validation SMAPE ({ensemble_smape_final:.2f}%)")
print(f"   4. If LB score ≈ Val SMAPE → Good generalization!")
print(f"   5. If LB score >> Val SMAPE → Possible overfitting")
print(f"   6. If good results → Proceed to Option D (CLIP fine-tuning)")

print("\n" + "=" * 80)
print("🎉 READY FOR SUBMISSION!")
print("=" * 80)
