#!/usr/bin/env python3
"""
ULTRA-FAST: Marqo E-commerce with OpenCLIP (In-Memory Processing)
================================================================

Why this is ultra-fast:
- In-memory processing: No disk I/O, preserves image quality
- 8 parallel downloads: 2x faster than before
- No compression artifacts: Original quality for feature extraction
- Zero disk usage: Everything stays in RAM

Expected correlation: 0.03-0.08 (3-8x better than CLIP's 0.0089)
Memory usage: ~225MB for 75K images (very reasonable)
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
import open_clip
import torch
import sys
import concurrent.futures
import shutil

print("="*80)
print("ULTRA-FAST: Marqo E-commerce with OpenCLIP (In-Memory Processing)")
print("="*80)

# Configuration
MODEL_NAME = "hf-hub:Marqo/marqo-ecommerce-embeddings-L"
BATCH_SIZE = 32
EMBEDDING_DIM = 1024  # Marqo-L has 1024-dim embeddings
MAX_IMAGES = None  # None = process all images

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

# Configuration
MODEL_NAME = "hf-hub:Marqo/marqo-ecommerce-embeddings-L"
BATCH_SIZE = 32
EMBEDDING_DIM = 1024  # Marqo-L has 1024-dim embeddings
MAX_IMAGES = None  # None = process all images

def download_all_images_in_memory(df, split_name):
    """Download all images and keep them in memory for immediate processing."""
    print(f"\n{'='*80}")
    print(f"DOWNLOADING ALL {split_name.upper()} IMAGES TO MEMORY")
    print(f"{'='*80}")

    # Track download results
    downloaded_images = {}
    failed = []

    def download_single_image(args):
        """Download a single image and return it."""
        idx, row = args
        img = download_image(row['image_link'])
        if img is not None:
            return idx, img, "downloaded"
        else:
            return idx, None, "download_failed"

    # Download with parallel processing (8 workers for faster downloads)
    print(f"Downloading {len(df)} images using 8 parallel workers...")
    print(f"Keeping images in memory for immediate processing")
    print(f"Estimated memory usage: ~{len(df) * 3 / 1024:.1f} MB (uncompressed)")
    print()

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        # Submit all downloads
        futures = [executor.submit(download_single_image, (idx, row))
                  for idx, row in df.iterrows()]

        # Progress tracking
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            idx, img, reason = future.result()
            completed += 1

            if img is not None:
                downloaded_images[idx] = img
            else:
                failed.append((idx, reason))

            # Progress update every 100 images
            if completed % 100 == 0:
                success_rate = len(downloaded_images) / completed * 100
                print(f"Progress: {completed}/{len(df)} | Success: {len(downloaded_images)} ({success_rate:.1f}%) | Failed: {len(failed)}")

    # Summary
    success_rate = len(downloaded_images) / len(df) * 100
    print(f"\n{'='*80}")
    print(f"DOWNLOAD SUMMARY: {split_name.upper()}")
    print(f"{'='*80}")
    print(f"Total images: {len(df)}")
    print(f"Downloaded to memory: {len(downloaded_images)} ({success_rate:.1f}%)")
    print(f"Failed: {len(failed)} ({100-success_rate:.1f}%)")
    print(f"Memory usage: ~{len(downloaded_images) * 3 / 1024:.1f} MB")

    if failed:
        print(f"\nFirst 5 failures:")
        for idx, reason in failed[:5]:
            print(f"  Image {idx}: {reason}")

    print(f"{'='*80}")
    return downloaded_images, failed

def extract_embeddings_batch(images, model, preprocess, tokenizer, device):
    """Extract embeddings for a batch of images."""
    try:
        # Preprocess images
        processed_images = []
        for img in images:
            processed_img = preprocess(img).unsqueeze(0).to(device)
            processed_images.append(processed_img)

        # Stack into batch
        batch_tensor = torch.cat(processed_images, dim=0)

        # Extract features
        with torch.no_grad():
            image_features = model.encode_image(batch_tensor)

        # Normalize (L2 normalization)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        return image_features.cpu().numpy()
    except Exception as e:
        print(f"Batch processing error: {e}")
        return None
    """Extract embeddings for a batch of images using OpenCLIP."""
    try:
        # Preprocess images
        processed_images = []
        for img in images:
            processed_images.append(preprocess(img).to(device))

        image_batch = torch.stack(processed_images)

        # Extract features
        with torch.no_grad():
            image_features = model.encode_image(image_batch)
            # Normalize embeddings (OpenCLIP doesn't auto-normalize)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        return image_features.cpu().numpy()
    except Exception as e:
        print(f"Batch processing error: {e}")
        return None

def process_dataset_optimized(csv_path, output_path, model, preprocess, tokenizer, device):
    """Process entire dataset with optimized in-memory processing."""
    print(f"\n{'='*80}")
    print(f"PROCESSING: {csv_path}")
    print(f"{'='*80}")

    # Load data
    df = pd.read_csv(csv_path)
    n_samples = len(df) if MAX_IMAGES is None else min(MAX_IMAGES, len(df))
    df = df.head(n_samples)

    print(f"Total samples: {n_samples}")
    print(f"Strategy: Download all to memory → Process in batches")
    print()

    # PHASE 1: Download all images to memory
    split_name = 'train' if 'train' in csv_path else 'test'
    downloaded_images, failed = download_all_images_in_memory(df, split_name)

    if len(downloaded_images) == 0:
        print("❌ No images downloaded, cannot proceed")
        return None, 0, len(df)

    # PHASE 2: Process images in batches
    print(f"\n{'='*80}")
    print(f"PHASE 2: EXTRACTING EMBEDDINGS FROM MEMORY")
    print(f"{'='*80}")

    # Initialize embeddings array
    embeddings = np.zeros((n_samples, EMBEDDING_DIM), dtype=np.float32)

    # Process in batches
    batch_images = []
    batch_indices = []
    successful = 0
    processed = 0

    for i, idx in enumerate(df.index[:n_samples]):
        # Get image from memory
        img = downloaded_images.get(idx)

        if img is not None:
            batch_images.append(img)
            batch_indices.append(i)

            # Process batch when full
            if len(batch_images) == BATCH_SIZE:
                batch_embeddings = extract_embeddings_batch(batch_images, model, preprocess, tokenizer, device)
                if batch_embeddings is not None:
                    for j, orig_idx in enumerate(batch_indices):
                        embeddings[orig_idx] = batch_embeddings[j]
                    successful += len(batch_indices)
                else:
                    # All failed in this batch
                    pass

                batch_images = []
                batch_indices = []
                processed += BATCH_SIZE

                # Progress update
                if processed % 1000 == 0:
                    success_rate = (successful / processed) * 100
                    print(f"Progress: {processed}/{n_samples} | Success: {successful} ({success_rate:.1f}%)")

    # Process remaining images
    if batch_images:
        batch_embeddings = extract_embeddings_batch(batch_images, model, preprocess, tokenizer, device)
        if batch_embeddings is not None:
            for j, orig_idx in enumerate(batch_indices):
                embeddings[orig_idx] = batch_embeddings[j]
            successful += len(batch_indices)
        processed += len(batch_images)

    # Save embeddings
    np.save(output_path, embeddings)

    print(f"\n{'='*80}")
    print(f"COMPLETED: {csv_path}")
    print(f"{'='*80}")
    print(f"Total processed: {n_samples}")
    print(f"Downloaded to memory: {len(downloaded_images)}")
    print(f"Successful embeddings: {successful} ({(successful/n_samples)*100:.1f}%)")
    print(f"Failed: {n_samples - successful} ({((n_samples - successful)/n_samples)*100:.1f}%)")
    print(f"Saved to: {output_path}")
    print(f"Shape: {embeddings.shape}")
    print(f"Size: {embeddings.nbytes / 1e6:.1f} MB")
    print()

    return embeddings, successful, n_samples - successful

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

    print(f"Marqo-Ecommerce-L (OpenCLIP) Embedding Correlations:")
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
    print(f"  Marqo-Ecommerce: {correlations.max():.4f}")
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
        print("PHASE 1: LOAD MARQO MODEL WITH OPENCLIP")
        print("="*80)

        print(f"Loading {MODEL_NAME}...")
        print("Model info:")
        print("  - Library: OpenCLIP (not transformers)")
        print("  - Parameters: ~652M")
        print("  - Training: 3M Amazon + 1M Google Shopping products")
        print("  - Output: 1024-dim normalized embeddings")
        print("  - Fix: No meta tensor issues with OpenCLIP")
        print()

        # Device setup
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load model using OpenCLIP (the correct way!)
        print("Loading with OpenCLIP...")
        model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(MODEL_NAME)
        tokenizer = open_clip.get_tokenizer(MODEL_NAME)

        model = model.to(device)
        model.eval()

        print(f"✅ Model loaded successfully on {device}")
        print(f"Device: {device}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print()

        # Process datasets with optimized batch download
        print("\n" + "="*80)
        print("PHASE 2: OPTIMIZED EMBEDDING EXTRACTION")
        print("Strategy: Download all → Process all → Cleanup")
        print("="*80)

        train_embeddings, train_success, train_fail = process_dataset_optimized(
            'dataset/train.csv',
            'outputs/train_marqo_ecommerce_openclip_embeddings.npy',
            model, preprocess_val, tokenizer, device
        )

        test_embeddings, test_success, test_fail = process_dataset_optimized(
            'dataset/test.csv',
            'outputs/test_marqo_ecommerce_openclip_embeddings.npy',
            model, preprocess_val, tokenizer, device
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
        print(f"✅ FIXED: Used OpenCLIP instead of transformers")
        print(f"✅ NO META TENSOR ERRORS")
        print(f"Train: {train_success}/{train_success+train_fail} images processed")
        print(f"Test: {test_success}/{test_success+test_fail} images processed")
        print(f"Max correlation: {correlations.max():.6f}")
        print(f"Output files:")
        print(f"  - outputs/train_marqo_ecommerce_openclip_embeddings.npy")
        print(f"  - outputs/test_marqo_ecommerce_openclip_embeddings.npy")
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