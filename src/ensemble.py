"""
Ensemble model combining LightGBM, Text MLP, and Image MLP.
Weighted averaging with optimized weights.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from scipy.optimize import minimize
import sys
sys.path.append('..')
from config.config import ENSEMBLE_WEIGHTS


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Symmetric Mean Absolute Percentage Error (SMAPE).
    
    SMAPE = 100 * mean(|y_pred - y_true| / (|y_true| + |y_pred|))
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        SMAPE score (0-100, lower is better)
    """
    # Clip predictions to avoid division by zero
    y_pred = np.clip(y_pred, 0.01, None)
    
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred))
    
    # Handle edge cases
    denominator = np.where(denominator == 0, 1e-8, denominator)
    
    smape_value = 100 * np.mean(numerator / denominator)
    
    return smape_value


class Ensemble:
    """Weighted ensemble of multiple models."""
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize ensemble with weights.
        
        Args:
            weights: Dict mapping model names to weights (should sum to 1.0)
        """
        if weights is None:
            weights = ENSEMBLE_WEIGHTS.copy()
        
        # Normalize weights to sum to 1
        total = sum(weights.values())
        self.weights = {k: v/total for k, v in weights.items()}
        
        print(f"Ensemble weights: {self.weights}")
    
    def predict(self, predictions_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Combine predictions from multiple models.
        
        Args:
            predictions_dict: Dict mapping model names to prediction arrays
            
        Returns:
            Weighted average predictions
        """
        # Validate all predictions have same length
        lengths = [len(preds) for preds in predictions_dict.values()]
        assert len(set(lengths)) == 1, "All predictions must have same length"
        
        ensemble_preds = np.zeros(lengths[0])
        
        for model_name, preds in predictions_dict.items():
            weight = self.weights.get(model_name, 0.0)
            ensemble_preds += weight * preds
        
        return ensemble_preds
    
    def optimize_weights(self, 
                        predictions_dict: Dict[str, np.ndarray],
                        y_true: np.ndarray) -> Dict[str, float]:
        """
        Optimize ensemble weights to minimize SMAPE.
        
        Args:
            predictions_dict: Dict of model predictions
            y_true: True values
            
        Returns:
            Optimized weights dict
        """
        model_names = list(predictions_dict.keys())
        n_models = len(model_names)
        
        # Convert predictions to matrix
        pred_matrix = np.column_stack([predictions_dict[name] for name in model_names])
        
        def objective(weights):
            """SMAPE loss for given weights."""
            ensemble_pred = pred_matrix @ weights
            return smape(y_true, ensemble_pred)
        
        # Constraints: weights sum to 1, all weights >= 0
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        ]
        bounds = [(0.0, 1.0) for _ in range(n_models)]
        
        # Initial guess: equal weights
        initial_weights = np.ones(n_models) / n_models
        
        # Optimize
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if result.success:
            optimized_weights = {name: weight for name, weight in zip(model_names, result.x)}
            print(f"Optimized weights: {optimized_weights}")
            print(f"Initial SMAPE: {objective(initial_weights):.4f}")
            print(f"Optimized SMAPE: {result.fun:.4f}")
            
            self.weights = optimized_weights
            return optimized_weights
        else:
            print("Optimization failed, keeping original weights")
            return self.weights


def cv_ensemble_predictions(lgb_models: List,
                            text_mlp_models: List,
                            image_mlp_models: List,
                            X_tabular: pd.DataFrame,
                            X_text_embeddings: np.ndarray,
                            X_image_embeddings: np.ndarray,
                            text_scalers: List,
                            image_scalers: List,
                            ensemble: Ensemble) -> np.ndarray:
    """
    Generate ensemble predictions by averaging CV fold models.
    
    Args:
        lgb_models: List of LightGBM models from CV
        text_mlp_models: List of text MLP models from CV
        image_mlp_models: List of image MLP models from CV
        X_tabular: Tabular features
        X_text_embeddings: Text embeddings
        X_image_embeddings: Image embeddings
        text_scalers: List of scalers for text MLP
        image_scalers: List of scalers for image MLP
        ensemble: Ensemble with weights
        
    Returns:
        Final ensemble predictions
    """
    from models import predict_lightgbm, predict_mlp
    
    n_folds = len(lgb_models)
    
    # Average predictions across folds
    lgb_preds = np.mean([predict_lightgbm(model, X_tabular) for model in lgb_models], axis=0)
    
    text_preds = np.mean([
        predict_mlp(model, X_text_embeddings, scaler) 
        for model, scaler in zip(text_mlp_models, text_scalers)
    ], axis=0)
    
    image_preds = np.mean([
        predict_mlp(model, X_image_embeddings, scaler)
        for model, scaler in zip(image_mlp_models, image_scalers)
    ], axis=0)
    
    # Combine with ensemble
    predictions_dict = {
        'lightgbm': lgb_preds,
        'text_mlp': text_preds,
        'image_mlp': image_preds
    }
    
    final_preds = ensemble.predict(predictions_dict)
    
    return final_preds


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model"):
    """
    Evaluate model performance with multiple metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        model_name: Name for logging
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    smape_score = smape(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n{model_name} Performance:")
    print(f"  SMAPE: {smape_score:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²: {r2:.4f}")
    
    return {
        'smape': smape_score,
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }


if __name__ == '__main__':
    # Test ensemble
    y_true = np.array([10.0, 20.0, 15.0, 30.0, 25.0])
    
    predictions = {
        'model1': np.array([9.5, 21.0, 14.5, 31.0, 24.0]),
        'model2': np.array([10.2, 19.5, 15.5, 29.0, 26.0]),
        'model3': np.array([9.8, 20.5, 15.0, 30.5, 25.5])
    }
    
    # Equal weights
    ensemble = Ensemble({'model1': 0.33, 'model2': 0.33, 'model3': 0.34})
    ensemble_pred = ensemble.predict(predictions)
    
    print(f"True: {y_true}")
    print(f"Ensemble: {ensemble_pred}")
    print(f"SMAPE: {smape(y_true, ensemble_pred):.4f}")
    
    # Optimize weights
    ensemble.optimize_weights(predictions, y_true)
