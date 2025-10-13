#!/usr/bin/env python3
"""
ULTRA-FAST: Marqo E-commerce with OpenCLIP (Multiprocessing Download)
===================================================================

Strategy: Multiprocessing download (100 processes) → Batch processing → Cleanup
- 100 parallel downloads (inspired by utils.py)
- PNG lossless quality for maximum feature extraction
- Memory-efficient batch processing (32 images at a time)
- Peak memory: ~1GB, Peak disk: ~11GB

Expected correlation: 0.03-0.08 (3-8x better than CLIP's 0.0089)
Download speed: Should be blazing fast with 100 processes!
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
print("ULTRA-FAST: Marqo E-commerce with OpenCLIP (Multiprocessing)")
print("="*80)

# Configuration
MODEL_NAME = "hf-hub:Marqo/marqo-ecommerce-embeddings-L"
BATCH_SIZE = 32
EMBEDDING_DIM = 1024  # Marqo-L has 1024-dim embeddings
MAX_IMAGES = None  # None = process all images

def download_image(url, timeout=5, max_retries=2):
    """Download image from URL with retry logic (improved version)."""
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

def download_single_image(idx, row, savefolder):
    """Download a single image (global function for multiprocessing pickle compatibility)."""
    image_link = row['image_link']
    filename = f"{idx}.png"  # Use PNG for lossless quality
    image_save_path = os.path.join(savefolder, filename)

    # Skip if already downloaded
    if os.path.exists(image_save_path):
        return f"SKIP: {idx}"

    # Download with retries (improved version)
    for attempt in range(3):  # 3 retries
        try:
            response = requests.get(image_link, timeout=10, stream=True)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content)).convert('RGB')
                img.save(image_save_path, 'PNG')  # Lossless PNG
                return f"SUCCESS: {idx}"
            else:
                continue
        except Exception as e:
            if attempt == 2:  # Last attempt
                return f"FAILED: {idx} - {str(e)}"
            continue

    return f"FAILED: {idx}"

def download_all_images_ultra_fast(df, image_dir, split_name):
    """Ultra-fast download using multiprocessing.Pool (inspired by utils.py)."""
    import multiprocessing
    from functools import partial

    print(f"\n{'='*80}")
    print(f"ULTRA-FAST DOWNLOAD: {split_name.upper()} IMAGES")
    print(f"{'='*80}")

    # Create image directory
    os.makedirs(image_dir, exist_ok=True)

    # Ultra-fast download using multiprocessing (reduced from 100 to 64 for stability)
    print(f"Downloading {len(df)} images using multiprocessing.Pool(64)...")
    print(f"Target directory: {image_dir}")
    print(f"Format: PNG (lossless, high quality)")
    print(f"Estimated space needed: ~{len(df) * 150 / 1024:.1f} MB")
    print("Starting download... (this may take a few minutes)")
    print()

    download_partial = partial(download_single_image, savefolder=image_dir)

    results = []
    try:
        with multiprocessing.Pool(processes=64) as pool:  # Reduced from 100 to 64
            print("🔄 Pool created, starting starmap...")
            # Use imap_unordered for better performance and progress tracking
            for result in pool.starmap(download_partial, df.iterrows()):
                results.append(result)

                # Progress update every 200 results (more frequent)
                if len(results) % 200 == 0:
                    success_count = sum(1 for r in results if r.startswith("SUCCESS"))
                    skip_count = sum(1 for r in results if r.startswith("SKIP"))
                    fail_count = sum(1 for r in results if r.startswith("FAILED"))
                    success_rate = (success_count / len(results)) * 100 if len(results) > 0 else 0
                    print(f"Progress: {len(results)}/{len(df)} | Success: {success_count} | Skip: {skip_count} | Fail: {fail_count} ({success_rate:.1f}%)")

            print("✅ Starmap completed")

    except Exception as e:
        print(f"❌ Multiprocessing error: {e}")
        print("🔄 Falling back to sequential download...")

        # Fallback to sequential download
        results = []
        for idx, row in df.iterrows():
            result = download_single_image(idx, row, image_dir)
            results.append(result)

            if len(results) % 100 == 0:
                success_count = sum(1 for r in results if r.startswith("SUCCESS"))
                skip_count = sum(1 for r in results if r.startswith("SKIP"))
                fail_count = sum(1 for r in results if r.startswith("FAILED"))
                success_rate = (success_count / len(results)) * 100 if len(results) > 0 else 0
                print(f"Fallback Progress: {len(results)}/{len(df)} | Success: {success_count} | Skip: {skip_count} | Fail: {fail_count} ({success_rate:.1f}%)")

    # Final summary
    success_count = sum(1 for r in results if r.startswith("SUCCESS"))
    skip_count = sum(1 for r in results if r.startswith("SKIP"))
    fail_count = sum(1 for r in results if r.startswith("FAILED"))

    success_rate = success_count / len(df) * 100 if len(df) > 0 else 0
    print(f"\n{'='*80}")
    print(f"DOWNLOAD SUMMARY: {split_name.upper()}")
    print(f"{'='*80}")
    print(f"Total images: {len(df)}")
    print(f"Downloaded: {success_count} ({success_rate:.1f}%)")
    print(f"Skipped (already exist): {skip_count}")
    print(f"Failed: {fail_count} ({100-success_rate:.1f}%)")
    print(f"Disk space used: ~{success_count * 150 / 1024:.1f} MB (PNG)")

    if fail_count > 0:
        print(f"\nFirst 5 failures:")
        failed_results = [r for r in results if r.startswith("FAILED")][:5]
        for result in failed_results:
            print(f"  {result}")

    print(f"{'='*80}")
    return success_count, fail_count

def load_image_from_disk(image_path):
    """Load image from local disk."""
    try:
        return Image.open(image_path).convert('RGB')
    except Exception as e:
        return None

def cleanup_image_directory(image_dir):
    """Remove all images from disk to free space."""
    print(f"\n🧹 Cleaning up: {image_dir}")
    if os.path.exists(image_dir):
        shutil.rmtree(image_dir)
        print("✅ Disk space freed")
    else:
        print("⚠️ Directory not found")

def extract_embeddings_batch(images, model, preprocess, tokenizer, device):
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

def process_dataset_hybrid(csv_path, output_path, model, preprocess, tokenizer, device):
    """Process dataset with hybrid approach: download to disk, then batch process."""
    print(f"\n{'='*80}")
    print(f"HYBRID PROCESSING: {csv_path}")
    print(f"{'='*80}")

    # Load data
    df = pd.read_csv(csv_path)
    n_samples = len(df) if MAX_IMAGES is None else min(MAX_IMAGES, len(df))
    df = df.head(n_samples)

    print(f"Total samples: {n_samples}")
    print(f"Strategy: Download all to disk → Process in batches → Cleanup")
    print()

    # Create temporary image directory
    split_name = 'train' if 'train' in csv_path else 'test'
    image_dir = f'/tmp/{split_name}_images'

    # PHASE 1: Ultra-fast download to disk
    downloaded_count, failed_count = download_all_images_ultra_fast(df, image_dir, split_name)

    if downloaded_count == 0:
        print("❌ No images downloaded, cannot proceed")
        return None, 0, len(df)

    # PHASE 2: Process in memory-efficient batches
    print(f"\n{'='*80}")
    print(f"PHASE 2: BATCH PROCESSING FROM DISK")
    print(f"{'='*80}")

    # Initialize embeddings array
    embeddings = np.zeros((n_samples, EMBEDDING_DIM), dtype=np.float32)

    # Process in batches (load from disk, process, discard from memory)
    successful = 0
    processed = 0

    for start_idx in range(0, n_samples, BATCH_SIZE):
        end_idx = min(start_idx + BATCH_SIZE, n_samples)
        batch_indices = list(range(start_idx, end_idx))

        # Load batch images from disk
        batch_images = []
        valid_indices = []

        for i in batch_indices:
            idx = df.index[i]
            image_path = os.path.join(image_dir, f"{idx}.png")
            img = load_image_from_disk(image_path)

            if img is not None:
                batch_images.append(img)
                valid_indices.append(i)

        # Process batch if we have images
        if batch_images:
            batch_embeddings = extract_embeddings_batch(batch_images, model, preprocess, tokenizer, device)
            if batch_embeddings is not None:
                for j, orig_idx in enumerate(valid_indices):
                    embeddings[orig_idx] = batch_embeddings[j]
                successful += len(valid_indices)

        processed += len(batch_indices)

        # Progress update
        if processed % 1000 == 0:
            success_rate = (successful / processed) * 100
            print(f"Progress: {processed}/{n_samples} | Success: {successful} ({success_rate:.1f}%)")

    # PHASE 3: Cleanup disk
    cleanup_image_directory(image_dir)

    # Save embeddings
    np.save(output_path, embeddings)

    print(f"\n{'='*80}")
    print(f"COMPLETED: {csv_path}")
    print(f"{'='*80}")
    print(f"Total processed: {n_samples}")
    print(f"Downloaded: {downloaded_count}")
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

        # Process datasets with hybrid approach
        print("\n" + "="*80)
        print("PHASE 2: HYBRID EMBEDDING EXTRACTION")
        print("Strategy: Aggressive download to disk → Batch processing → Cleanup")
        print("="*80)

        train_embeddings, train_success, train_fail = process_dataset_hybrid(
            'dataset/train.csv',
            'outputs/train_marqo_ecommerce_openclip_embeddings.npy',
            model, preprocess_val, tokenizer, device
        )

        test_embeddings, test_success, test_fail = process_dataset_hybrid(
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