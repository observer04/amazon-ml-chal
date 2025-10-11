"""
Model implementations: LightGBM and MLP for text/image embeddings.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Optional, Dict
import sys
sys.path.append('..')
from config.config import (
    LIGHTGBM_PARAMS, MLP_HIDDEN_DIMS, DROPOUT_RATE, LEARNING_RATE,
    BATCH_SIZE, EPOCHS, EARLY_STOPPING_PATIENCE, N_SPLITS, RANDOM_STATE,
    TEXT_EMBEDDING_DIM, IMAGE_EMBEDDING_DIM
)


# ===== LightGBM Model =====

def train_lightgbm(X_train: pd.DataFrame, 
                   y_train: pd.Series,
                   X_val: Optional[pd.DataFrame] = None,
                   y_val: Optional[pd.Series] = None,
                   params: Optional[Dict] = None) -> lgb.Booster:
    """
    Train LightGBM model on tabular features.
    
    Args:
        X_train: Training features
        y_train: Training targets (prices)
        X_val: Validation features (optional)
        y_val: Validation targets (optional)
        params: LightGBM parameters (optional)
        
    Returns:
        Trained LightGBM booster
    """
    if params is None:
        params = LIGHTGBM_PARAMS.copy()
    
    # Create datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    
    valid_sets = [train_data]
    valid_names = ['train']
    
    if X_val is not None and y_val is not None:
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        valid_sets.append(val_data)
        valid_names.append('valid')
    
    # Train
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=100)
        ]
    )
    
    return model


def predict_lightgbm(model: lgb.Booster, X: pd.DataFrame) -> np.ndarray:
    """
    Generate predictions with LightGBM model.
    
    Args:
        model: Trained LightGBM booster
        X: Features
        
    Returns:
        Predictions
    """
    return model.predict(X, num_iteration=model.best_iteration)


# ===== MLP Model =====

class EmbeddingDataset(Dataset):
    """PyTorch Dataset for embeddings and targets."""
    
    def __init__(self, embeddings: np.ndarray, targets: Optional[np.ndarray] = None):
        self.embeddings = torch.FloatTensor(embeddings)
        self.targets = torch.FloatTensor(targets) if targets is not None else None
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        if self.targets is not None:
            return self.embeddings[idx], self.targets[idx]
        else:
            return self.embeddings[idx]


class MLP(nn.Module):
    """Multi-layer perceptron for regression."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = None, dropout: float = DROPOUT_RATE):
        super(MLP, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = MLP_HIDDEN_DIMS
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x).squeeze(-1)


def train_mlp(embeddings_train: np.ndarray,
              targets_train: np.ndarray,
              embeddings_val: Optional[np.ndarray] = None,
              targets_val: Optional[np.ndarray] = None,
              input_dim: Optional[int] = None,
              epochs: int = EPOCHS,
              batch_size: int = BATCH_SIZE,
              lr: float = LEARNING_RATE,
              device: Optional[str] = None) -> Tuple[MLP, StandardScaler]:
    """
    Train MLP model on embeddings.
    
    Args:
        embeddings_train: Training embeddings
        targets_train: Training targets (prices)
        embeddings_val: Validation embeddings (optional)
        targets_val: Validation targets (optional)
        input_dim: Input dimension (auto-detected if None)
        epochs: Number of epochs
        batch_size: Batch size
        lr: Learning rate
        device: 'cuda', 'cpu', or None (auto-detect)
        
    Returns:
        Tuple of (trained model, scaler for targets)
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if input_dim is None:
        input_dim = embeddings_train.shape[1]
    
    # Scale targets (log transform for price)
    scaler = StandardScaler()
    targets_train_scaled = scaler.fit_transform(np.log1p(targets_train).reshape(-1, 1)).flatten()
    
    # Create datasets
    train_dataset = EmbeddingDataset(embeddings_train, targets_train_scaled)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    if embeddings_val is not None and targets_val is not None:
        targets_val_scaled = scaler.transform(np.log1p(targets_val).reshape(-1, 1)).flatten()
        val_dataset = EmbeddingDataset(embeddings_val, targets_val_scaled)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    else:
        val_loader = None
    
    # Initialize model
    model = MLP(input_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for embeddings, targets in train_loader:
            embeddings = embeddings.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(embeddings)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for embeddings, targets in val_loader:
                    embeddings = embeddings.to(device)
                    targets = targets.to(device)
                    
                    outputs = model(embeddings)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                model.load_state_dict(best_model_state)
                break
        else:
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}")
    
    return model, scaler


def predict_mlp(model: MLP, 
                embeddings: np.ndarray,
                scaler: StandardScaler,
                device: Optional[str] = None) -> np.ndarray:
    """
    Generate predictions with MLP model.
    
    Args:
        model: Trained MLP model
        embeddings: Input embeddings
        scaler: Target scaler (for inverse transform)
        device: 'cuda', 'cpu', or None (auto-detect)
        
    Returns:
        Predictions (original scale)
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model.eval()
    dataset = EmbeddingDataset(embeddings)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    predictions_scaled = []
    
    with torch.no_grad():
        for embeddings_batch in loader:
            embeddings_batch = embeddings_batch.to(device)
            outputs = model(embeddings_batch)
            predictions_scaled.append(outputs.cpu().numpy())
    
    predictions_scaled = np.concatenate(predictions_scaled)
    
    # Inverse transform
    predictions = np.expm1(scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten())
    
    return predictions


# ===== Cross-Validation =====

def cv_train_lightgbm(X: pd.DataFrame, y: pd.Series, n_splits: int = N_SPLITS) -> Tuple[List[lgb.Booster], np.ndarray]:
    """
    Train LightGBM with K-fold cross-validation.
    
    Returns:
        Tuple of (list of models, OOF predictions)
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    models = []
    oof_preds = np.zeros(len(X))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\n=== Fold {fold + 1}/{n_splits} ===")
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = train_lightgbm(X_train, y_train, X_val, y_val)
        models.append(model)
        
        oof_preds[val_idx] = predict_lightgbm(model, X_val)
    
    return models, oof_preds


def cv_train_mlp(embeddings: np.ndarray, targets: np.ndarray, n_splits: int = N_SPLITS) -> Tuple[List[MLP], List[StandardScaler], np.ndarray]:
    """
    Train MLP with K-fold cross-validation.
    
    Returns:
        Tuple of (list of models, list of scalers, OOF predictions)
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    models = []
    scalers = []
    oof_preds = np.zeros(len(embeddings))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(embeddings)):
        print(f"\n=== Fold {fold + 1}/{n_splits} ===")
        
        emb_train, emb_val = embeddings[train_idx], embeddings[val_idx]
        y_train, y_val = targets[train_idx], targets[val_idx]
        
        model, scaler = train_mlp(emb_train, y_train, emb_val, y_val)
        models.append(model)
        scalers.append(scaler)
        
        oof_preds[val_idx] = predict_mlp(model, emb_val, scaler)
    
    return models, scalers, oof_preds


if __name__ == '__main__':
    # Test models with dummy data
    print("Testing LightGBM...")
    X_dummy = pd.DataFrame(np.random.randn(1000, 10))
    y_dummy = pd.Series(np.random.uniform(5, 50, 1000))
    
    lgb_model = train_lightgbm(X_dummy[:800], y_dummy[:800], X_dummy[800:], y_dummy[800:])
    lgb_preds = predict_lightgbm(lgb_model, X_dummy[800:])
    print(f"LightGBM predictions: {lgb_preds[:5]}")
    
    print("\nTesting MLP...")
    emb_dummy = np.random.randn(1000, 384)
    mlp_model, mlp_scaler = train_mlp(emb_dummy[:800], y_dummy[:800], emb_dummy[800:], y_dummy[800:], epochs=10)
    mlp_preds = predict_mlp(mlp_model, emb_dummy[800:], mlp_scaler)
    print(f"MLP predictions: {mlp_preds[:5]}")
