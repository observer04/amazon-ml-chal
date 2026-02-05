import numpy as np
import pandas as pd

# Load embeddings
print('Loading image embeddings...')
img_emb = np.load('outputs/train_image_embeddings_clip.npy')
print(f'Shape: {img_emb.shape}')
print(f'Mean: {img_emb.mean():.4f}')
print(f'Std: {img_emb.std():.4f}')
print(f'Min: {img_emb.min():.4f}')
print(f'Max: {img_emb.max():.4f}')
print()

# Check for issues
print('Checking for problems:')
print(f'NaN values: {np.isnan(img_emb).sum()}')
print(f'Inf values: {np.isinf(img_emb).sum()}')
print(f'All zeros rows: {(img_emb.sum(axis=1) == 0).sum()}')
print()

# Check variance
variances = img_emb.var(axis=0)
print(f'Feature variance: mean={variances.mean():.4f}, min={variances.min():.6f}, max={variances.max():.4f}')
print(f'Low variance features (<0.001): {(variances < 0.001).sum()} / 512')
print()

# Load prices
train_df = pd.read_csv('dataset/train_with_features.csv')
y_train = train_df['price'].values
print(f'Price statistics:')
print(f'  Mean: ${y_train.mean():.2f}')
print(f'  Median: ${np.median(y_train):.2f}')
print(f'  Min: ${y_train.min():.2f}')
print(f'  Max: ${y_train.max():.2f}')
print(f'  Std: ${y_train.std():.2f}')
print()

# Distribution
print('Price distribution:')
print(f'  <$10: {(y_train < 10).sum()} ({100*(y_train < 10).mean():.1f}%)')
print(f'  $10-$25: {((y_train >= 10) & (y_train < 25)).sum()} ({100*((y_train >= 10) & (y_train < 25)).mean():.1f}%)')
print(f'  $25-$50: {((y_train >= 25) & (y_train < 50)).sum()} ({100*((y_train >= 25) & (y_train < 50)).mean():.1f}%)')
print(f'  $50-$100: {((y_train >= 50) & (y_train < 100)).sum()} ({100*((y_train >= 50) & (y_train < 100)).mean():.1f}%)')
print(f'  >$100: {(y_train >= 100).sum()} ({100*(y_train >= 100).mean():.1f}%)')
print()

# Check correlation between embeddings and price
print('Analyzing image-price relationship:')
y_log = np.log1p(y_train)
print(f'Log-price range: [{y_log.min():.3f}, {y_log.max():.3f}]')
print()

# Sample some predictions
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

X_train, X_val, y_train_split, y_val = train_test_split(
    img_emb, y_log, test_size=0.2, random_state=42
)

# Quick linear model
model = Ridge(alpha=1.0)
model.fit(X_train, y_train_split)
val_pred = model.predict(X_val)
val_pred_price = np.expm1(val_pred)
val_actual_price = np.expm1(y_val)

from sklearn.metrics import r2_score
r2 = r2_score(y_val, val_pred)
smape = 100 * np.mean(2 * np.abs(val_pred_price - val_actual_price) / 
                      (np.abs(val_pred_price) + np.abs(val_actual_price)))

print(f'Simple Ridge regression:')
print(f'  R² (log-space): {r2:.4f}')
print(f'  SMAPE: {smape:.2f}%')
print(f'  Pred range: [${val_pred_price.min():.2f}, ${val_pred_price.max():.2f}]')
