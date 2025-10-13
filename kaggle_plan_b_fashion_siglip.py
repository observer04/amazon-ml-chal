#!/usr/bin/env python3
"""
PLAN B v2: Extract Marqo-FashionSigLIP embeddings
==================================================

Why this model:
- 400M parameters (well under 8B limit)
- Trained on 1M+ fashion/e-commerce products
- SigLIP-based (proven architecture without meta tensor issues)
- Used in production by leboncoin marketplace
- 768-dim embeddings optimized for product similarity

Fallback from: Marqo-ecommerce-L (meta tensor loading issues)
Alternative to: ResNet50 (too generic, not e-commerce specific)

Expected correlation: 0.03-0.08 (3-8x better than CLIP's 0.0089)
"""

import os
import warnings
warnings.filterwarnings('ignore')

# Suppress specific warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

import pandas as pd
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import torch
from transformers import AutoModel, AutoProcessor
from tqdm import tqdm
import sys

print("="*80)
print("PLAN B v2: Marqo-FashionSigLIP Embedding Extraction")
print("="*80)

# Configuration
MODEL_NAME = "Marqo/marqo-fashionSigLIP"
BATCH_SIZE = 32
EMBEDDING_DIM = 768
MAX_IMAGES = None  # None = process all images

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print()

def download_image(url, timeout=5, max_retries=2):
    """Download image from URL with retry logic."""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=timeout, stream=True)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content)).convert('RGB')
                return img
        except Exception as e:
            if attempt == max_retries - 1:
                return None
            continue
    return None

def extract_embeddings_batch(images, model, processor, device):
    """Extract embeddings for a batch of images."""
    try:
        # Process images
        inputs = processor(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Extract features
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
            # Normalize embeddings
            embeddings = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return embeddings.cpu().numpy()
    except Exception as e:
        print(f"Batch processing error: {e}")
        return None

def process_dataset(csv_path, output_path, model, processor, device):
    """Process entire dataset and extract embeddings."""
    print(f"\n{'='*80}")
    print(f"Processing: {csv_path}")
    print(f"{'='*80}")
    
    # Load data
    df = pd.read_csv(csv_path)
    n_samples = len(df) if MAX_IMAGES is None else min(MAX_IMAGES, len(df))
    df = df.head(n_samples)
    
    print(f"Total samples: {n_samples}")
    print()
    
    # Initialize embeddings array
    embeddings = np.zeros((n_samples, EMBEDDING_DIM), dtype=np.float32)
    
    # Process in batches
    batch_images = []
    batch_indices = []
    successful = 0
    failed = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting embeddings"):
        # Download image
        img = download_image(row['image_link'])
        
        if img is not None:
            batch_images.append(img)
            batch_indices.append(idx)
            
            # Process batch when full
            if len(batch_images) == BATCH_SIZE:
                batch_embeddings = extract_embeddings_batch(batch_images, model, processor, device)
                if batch_embeddings is not None:
                    for i, orig_idx in enumerate(batch_indices):
                        embeddings[orig_idx] = batch_embeddings[i]
                    successful += len(batch_indices)
                else:
                    failed += len(batch_indices)
                
                batch_images = []
                batch_indices = []
        else:
            failed += 1
        
        # Progress update every 1000 images
        if (idx + 1) % 1000 == 0:
            success_rate = (successful / (idx + 1)) * 100
            print(f"Progress: {idx+1}/{n_samples} | Success: {successful} ({success_rate:.1f}%) | Failed: {failed}")
    
    # Process remaining images
    if batch_images:
        batch_embeddings = extract_embeddings_batch(batch_images, model, processor, device)
        if batch_embeddings is not None:
            for i, orig_idx in enumerate(batch_indices):
                embeddings[orig_idx] = batch_embeddings[i]
            successful += len(batch_indices)
        else:
            failed += len(batch_indices)
    
    # Save embeddings
    np.save(output_path, embeddings)
    
    print(f"\n{'='*80}")
    print(f"COMPLETED: {csv_path}")
    print(f"{'='*80}")
    print(f"Total processed: {n_samples}")
    print(f"Successful: {successful} ({(successful/n_samples)*100:.1f}%)")
    print(f"Failed: {failed} ({(failed/n_samples)*100:.1f}%)")
    print(f"Saved to: {output_path}")
    print(f"Shape: {embeddings.shape}")
    print(f"Size: {embeddings.nbytes / 1e6:.1f} MB")
    print()
    
    return embeddings, successful, failed

def compute_correlations(train_embeddings, train_csv):
    """Compute correlation between embeddings and target price."""
    print(f"\n{'='*80}")
    print("CORRELATION ANALYSIS")
    print(f"{'='*80}")
    
    # Load target prices
    df = pd.read_csv(train_csv)
    prices = df['PRODUCT_LENGTH'].values[:len(train_embeddings)]
    
    print(f"Price statistics:")
    print(f"  Mean: {prices.mean():.2f}")
    print(f"  Std: {prices.std():.2f}")
    print(f"  Min: {prices.min():.2f}")
    print(f"  Max: {prices.max():.2f}")
    print()
    
    # Compute correlations for each embedding dimension
    correlations = []
    for i in range(train_embeddings.shape[1]):
        corr = np.corrcoef(train_embeddings[:, i], prices)[0, 1]
        if not np.isnan(corr):
            correlations.append(abs(corr))
        else:
            correlations.append(0)
    
    correlations = np.array(correlations)
    
    print(f"FashionSigLIP Embedding Correlations:")
    print(f"  Max correlation: {correlations.max():.6f}")
    print(f"  Mean correlation: {correlations.mean():.6f}")
    print(f"  Median correlation: {np.median(correlations):.6f}")
    print(f"  Dims with |corr| > 0.05: {(correlations > 0.05).sum()}")
    print(f"  Dims with |corr| > 0.02: {(correlations > 0.02).sum()}")
    print()
    
    # Compare to baselines
    print("COMPARISON TO BASELINES:")
    print(f"  CLIP baseline: 0.0089 (max correlation)")
    print(f"  Text baseline: 0.1089 (max correlation)")
    print(f"  FashionSigLIP: {correlations.max():.4f}")
    print()
    
    improvement_vs_clip = ((correlations.max() - 0.0089) / 0.0089) * 100
    print(f"Improvement vs CLIP: {improvement_vs_clip:+.1f}%")
    
    # Verdict
    print(f"\n{'='*80}")
    print("VERDICT:")
    print(f"{'='*80}")
    if correlations.max() > 0.05:
        print("✅ STRONG SIGNAL - Create training script with image features")
        print("   Expected SMAPE improvement: 5-8%")
    elif correlations.max() > 0.02:
        print("⚠️  MODERATE SIGNAL - Test in ensemble")
        print("   Expected SMAPE improvement: 2-4%")
    else:
        print("❌ WEAK SIGNAL - Consider alternative approaches")
        print("   Improvement uncertain, try other methods")
    print(f"{'='*80}")
    print()
    
    return correlations

def main():
    try:
        print("\n" + "="*80)
        print("PHASE 1: MODEL LOADING")
        print("="*80)
        
        print(f"Loading {MODEL_NAME}...")
        print("Model info:")
        print("  - Type: SigLIP vision encoder")
        print("  - Parameters: ~400M")
        print("  - Training: 1M+ fashion/e-commerce products")
        print("  - Output: 768-dim normalized embeddings")
        print()
        
        # Load model and processor
        model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
        model = model.to(device)
        model.eval()
        
        print(f"✅ Model loaded successfully")
        print()
        
        # Process datasets
        print("\n" + "="*80)
        print("PHASE 2: EMBEDDING EXTRACTION")
        print("="*80)
        
        train_embeddings, train_success, train_fail = process_dataset(
            'dataset/train.csv',
            'outputs/train_fashion_siglip_embeddings.npy',
            model, processor, device
        )
        
        test_embeddings, test_success, test_fail = process_dataset(
            'dataset/test.csv',
            'outputs/test_fashion_siglip_embeddings.npy',
            model, processor, device
        )
        
        # Correlation analysis
        print("\n" + "="*80)
        print("PHASE 3: CORRELATION ANALYSIS")
        print("="*80)
        correlations = compute_correlations(train_embeddings, 'dataset/train.csv')
        
        # Summary
        print("\n" + "="*80)
        print("EXECUTION SUMMARY")
        print("="*80)
        print(f"✅ Successfully completed all phases")
        print(f"Train: {train_success}/{train_success+train_fail} images processed")
        print(f"Test: {test_success}/{test_success+test_fail} images processed")
        print(f"Max correlation: {correlations.max():.6f}")
        print(f"Output files:")
        print(f"  - outputs/train_fashion_siglip_embeddings.npy")
        print(f"  - outputs/test_fashion_siglip_embeddings.npy")
        print("="*80)
        
    except Exception as e:
        print("\n" + "="*80)
        print("❌ FAILED")
        print("="*80)
        print(f"Error: {str(e)}")
        print()
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
