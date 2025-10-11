"""
Phase 3: Text Embeddings with Sentence-Transformers
===================================================

Generate text embeddings from catalog_content using
sentence-transformers (all-MiniLM-L6-v2).

Output:
- Train embeddings: 75,000 x 384 dims = ~115MB
- Test embeddings: 75,000 x 384 dims = ~115MB
- Total: ~230MB

Expected improvement: 62% → 52-57% SMAPE
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
from tqdm import tqdm
import torch
import gc


class TextEmbeddingExtractor:
    """Extract sentence embeddings using SentenceTransformer."""
    
    def __init__(self, model_name='all-MiniLM-L6-v2', device='cuda'):
        """
        Initialize sentence transformer model.
        
        Args:
            model_name: HuggingFace model name
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Using device: {self.device}")
        
        print(f"📥 Loading {model_name}...")
        self.model = SentenceTransformer(model_name, device=str(self.device))
        
        print(f"✅ Model loaded: {model_name}")
        print(f"   Output dimensions: {self.model.get_sentence_embedding_dimension()}")
    
    def extract_embeddings(self, texts, batch_size=64, save_path=None):
        """
        Extract embeddings for all texts.
        
        Args:
            texts: List of text strings
            batch_size: Batch size for encoding
            save_path: Path to save embeddings (optional)
            
        Returns:
            embeddings: (N, 384) numpy array
        """
        print(f"\n🔄 Processing {len(texts):,} texts in batches of {batch_size}...")
        
        # Handle missing/NaN values
        texts_clean = []
        for text in texts:
            if pd.isna(text) or text == '':
                texts_clean.append('unknown product')  # Default text for missing
            else:
                texts_clean.append(str(text))
        
        # Extract embeddings with progress bar
        embeddings = self.model.encode(
            texts_clean,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 normalization for cosine similarity
        )
        
        print(f"✅ Embeddings extracted: {embeddings.shape}")
        print(f"   Mean norm: {np.linalg.norm(embeddings, axis=1).mean():.3f}")
        print(f"   Missing/empty texts: {sum(pd.isna(t) or t == '' for t in texts):,}")
        
        # Save if path provided
        if save_path:
            np.save(save_path, embeddings)
            print(f"💾 Saved to: {save_path}")
            
            # Print file size
            file_size = Path(save_path).stat().st_size / (1024**2)
            print(f"   File size: {file_size:.1f} MB")
        
        return embeddings


def main():
    """Main execution pipeline."""
    print("="*70)
    print("PHASE 3: TEXT EMBEDDINGS")
    print("="*70)
    
    # Load datasets
    print("\n📂 Loading datasets...")
    train_df = pd.read_csv('dataset/train_with_features.csv')
    test_df = pd.read_csv('dataset/test_with_features.csv')
    
    print(f"✅ Train: {len(train_df):,} samples")
    print(f"✅ Test: {len(test_df):,} samples")
    
    # Create output directory
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)
    
    # Initialize extractor
    print("\n🔧 Initializing sentence-transformer...")
    extractor = TextEmbeddingExtractor(
        model_name='all-MiniLM-L6-v2',
        device='cuda'
    )
    
    # Extract train embeddings
    print("\n" + "="*70)
    print("EXTRACTING TRAIN TEXT EMBEDDINGS")
    print("="*70)
    
    train_embeddings = extractor.extract_embeddings(
        train_df['catalog_content'].tolist(),
        batch_size=64,
        save_path='outputs/train_text_embeddings.npy'
    )
    
    # Extract test embeddings
    print("\n" + "="*70)
    print("EXTRACTING TEST TEXT EMBEDDINGS")
    print("="*70)
    
    test_embeddings = extractor.extract_embeddings(
        test_df['catalog_content'].tolist(),
        batch_size=64,
        save_path='outputs/test_text_embeddings.npy'
    )
    
    # Summary
    print("\n\n" + "="*70)
    print("✅ PHASE 3 COMPLETE - TEXT EMBEDDINGS EXTRACTED")
    print("="*70)
    
    print(f"\n📊 Final Status:")
    print(f"   Train embeddings: {train_embeddings.shape}")
    print(f"   Test embeddings: {test_embeddings.shape}")
    
    # Calculate total disk usage
    train_size = Path('outputs/train_text_embeddings.npy').stat().st_size / (1024**2)
    test_size = Path('outputs/test_text_embeddings.npy').stat().st_size / (1024**2)
    total_size = train_size + test_size
    
    print(f"\n💾 Disk usage:")
    print(f"   Train embeddings: {train_size:.1f} MB")
    print(f"   Test embeddings: {test_size:.1f} MB")
    print(f"   Total: {total_size:.1f} MB")
    
    print(f"\n✅ SUCCESS: All text embeddings extracted")
    print(f"   Ready for model training!")
    
    print(f"\n🚀 Next: Phase 4 - Train multi-modal ensemble")
    print(f"   Branches:")
    print(f"   1. LightGBM on engineered features (30% weight)")
    print(f"   2. MLP on text embeddings (20% weight)")
    print(f"   3. MLP on image embeddings (50% weight)")


if __name__ == '__main__':
    main()
