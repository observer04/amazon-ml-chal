"""
Text embedding generation using sentence-transformers.
Converts catalog_content text to 384-dimensional vectors.
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Optional
import torch
import sys
sys.path.append('..')
from config.config import TEXT_MODEL_NAME, TEXT_EMBEDDING_DIM


class TextEmbedder:
    """Wrapper for sentence-transformers model."""
    
    def __init__(self, model_name: str = TEXT_MODEL_NAME, device: Optional[str] = None):
        """
        Initialize the text embedder.
        
        Args:
            model_name: HuggingFace model name
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"Loading text embedding model: {model_name} on {device}")
        self.model = SentenceTransformer(model_name, device=device)
        self.device = device
        self.embedding_dim = TEXT_EMBEDDING_DIM
        
    def encode(self, texts: List[str], batch_size: int = 256, show_progress: bool = True) -> np.ndarray:
        """
        Encode texts to embeddings.
        
        Args:
            texts: List of text strings
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            
        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        # Handle None/NaN values
        texts_clean = [str(text) if pd.notna(text) else '' for text in texts]
        
        embeddings = self.model.encode(
            texts_clean,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def encode_dataframe(self, df: pd.DataFrame, text_column: str = 'catalog_content') -> np.ndarray:
        """
        Encode text column from DataFrame.
        
        Args:
            df: DataFrame with text column
            text_column: Name of text column
            
        Returns:
            numpy array of embeddings
        """
        print(f"Encoding {len(df)} texts from column '{text_column}'...")
        return self.encode(df[text_column].tolist())


def create_text_embeddings(df: pd.DataFrame, 
                          text_column: str = 'catalog_content',
                          embedder: Optional[TextEmbedder] = None,
                          save_path: Optional[str] = None) -> np.ndarray:
    """
    Create text embeddings for entire DataFrame.
    
    Args:
        df: DataFrame with text column
        text_column: Name of column to embed
        embedder: Pre-initialized TextEmbedder (optional)
        save_path: Path to save embeddings (optional)
        
    Returns:
        numpy array of shape (len(df), 384)
    """
    if embedder is None:
        embedder = TextEmbedder()
    
    embeddings = embedder.encode_dataframe(df, text_column)
    
    if save_path:
        print(f"Saving embeddings to {save_path}")
        np.save(save_path, embeddings)
    
    print(f"Created embeddings with shape: {embeddings.shape}")
    return embeddings


def create_combined_text_embeddings(df: pd.DataFrame,
                                   embedder: Optional[TextEmbedder] = None) -> np.ndarray:
    """
    Create embeddings from multiple text fields combined.
    Combines item_name, bullets, and description with intelligent weighting.
    
    Args:
        df: DataFrame with parsed text features
        embedder: Pre-initialized TextEmbedder
        
    Returns:
        numpy array of embeddings
    """
    if embedder is None:
        embedder = TextEmbedder()
    
    # Combine text fields with priorities
    combined_texts = []
    for _, row in df.iterrows():
        parts = []
        
        # Item name (highest priority)
        if pd.notna(row.get('item_name', '')) and row.get('item_name', '').strip():
            parts.append(row['item_name'])
        
        # Bullets (medium priority)
        for i in range(1, 7):  # bullet_1 to bullet_6
            bullet_col = f'bullet_{i}'
            if bullet_col in row and pd.notna(row[bullet_col]) and row[bullet_col].strip():
                parts.append(row[bullet_col])
        
        # Description (lower priority)
        if pd.notna(row.get('description', '')) and row.get('description', '').strip():
            parts.append(row['description'])
        
        # Fallback to full catalog_content if nothing parsed
        if not parts and 'catalog_content' in row:
            combined = str(row['catalog_content']) if pd.notna(row['catalog_content']) else ''
        else:
            combined = ' | '.join(parts)
        
        combined_texts.append(combined)
    
    print(f"Encoding {len(combined_texts)} combined text fields...")
    embeddings = embedder.encode(combined_texts)
    
    return embeddings


if __name__ == '__main__':
    # Test encoding
    test_texts = [
        "Premium Organic Coffee Beans - 12 Ounce - Dark Roast",
        "Natural Honey - 16 oz - Raw and Unfiltered",
        "Gourmet Chocolate Bar - Artisan Crafted"
    ]
    
    embedder = TextEmbedder()
    embeddings = embedder.encode(test_texts, show_progress=False)
    
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Sample embedding (first 10 dims): {embeddings[0][:10]}")
    
    # Test similarity
    from sklearn.metrics.pairwise import cosine_similarity
    sim_matrix = cosine_similarity(embeddings)
    print(f"\nSimilarity matrix:")
    print(sim_matrix)
