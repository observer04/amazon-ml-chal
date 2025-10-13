#!/usr/bin/env python3
"""
KAGGLE PHASE 1: Extract Marqo E-commerce Image Embeddings
==========================================================
Single monolithic script to run on Kaggle GPU notebook.

Marqo/marqo-ecommerce-embeddings-L:
- 652M params, 1024-dim embeddings
- Trained on 3M Amazon products + 1M Google Shopping
- 38.9% better than Amazon-Titan on Amazon product tasks

This script:
1. Installs dependencies
2. Downloads Marqo model
3. Processes all 75K train + 75K test images
4. Saves embeddings as .npy files
5. Computes correlations with price
6. Writes detailed results to KAGGLE_RUN_RESULTS.md

Usage on Kaggle:
    python kaggle_extract_marqo_embeddings.py > KAGGLE_RUN_RESULTS.md 2>&1
"""

import os
import sys
import time
import subprocess
from datetime import datetime
import traceback

# Redirect all output to both console and file
class TeeOutput:
    def __init__(self, *files):
        self.files = files
    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

# Open results file and tee output
results_file = open('/kaggle/working/KAGGLE_RUN_RESULTS.md', 'w')
original_stdout = sys.stdout
original_stderr = sys.stderr
sys.stdout = TeeOutput(sys.stdout, results_file)
sys.stderr = TeeOutput(sys.stderr, results_file)

# Print markdown header
print("# KAGGLE RUN RESULTS: Marqo Embedding Extraction")
print(f"**Started:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"**Script:** kaggle_extract_marqo_embeddings.py")
print()
print("```")

# Phase 1: Setup and Installation
print("="*80)
print("PHASE 1: SETUP")
print("="*80)
start_time = time.time()

def install_packages():
    """Install required packages"""
    packages = [
        'transformers',
        'torch',
        'torchvision', 
        'pillow',
        'accelerate',
        'sentencepiece',
        'protobuf',
        'scipy'
    ]
    
    print("\nInstalling packages...")
    for pkg in packages:
        print(f"  Installing {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
    print("✓ All packages installed")

try:
    install_packages()
except Exception as e:
    print(f"✗ Installation failed: {e}")
    sys.exit(1)

# Import after installation
import torch
import numpy as np
import pandas as pd
from PIL import Image
import requests
from io import BytesIO
from transformers import AutoModel, AutoProcessor
from scipy.stats import pearsonr
from tqdm import tqdm

print(f"\n✓ Setup complete in {time.time() - start_time:.1f}s")

# Phase 2: Load Data
print("\n" + "="*80)
print("PHASE 2: LOAD DATA")
print("="*80)

try:
    df_train = pd.read_csv('/kaggle/input/amazon-ml-challenge-2024/train.csv')
    df_test = pd.read_csv('/kaggle/input/amazon-ml-challenge-2024/test.csv')
    print(f"✓ Train: {len(df_train):,} samples")
    print(f"✓ Test: {len(df_test):,} samples")
except Exception as e:
    print(f"✗ Failed to load data: {e}")
    sys.exit(1)

# Phase 3: Load Marqo Model
print("\n" + "="*80)
print("PHASE 3: LOAD MARQO MODEL")
print("="*80)

model_start = time.time()

try:
    model_name = 'Marqo/marqo-ecommerce-embeddings-L'
    print(f"Loading {model_name}...")
    print("This may take 2-3 minutes (652M params)...")
    
    # Load model without device_map to avoid accelerate issues
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    
    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model loaded on {device}")
    print(f"✓ Load time: {time.time() - model_start:.1f}s")
    
except Exception as e:
    print(f"✗ Model loading failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Phase 4: Extract Embeddings
print("\n" + "="*80)
print("PHASE 4: EXTRACT IMAGE EMBEDDINGS")
print("="*80)

def extract_embeddings(df, split_name):
    """Extract embeddings for all images in dataframe"""
    embeddings = []
    failed_indices = []
    
    print(f"\nProcessing {len(df)} {split_name} images...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"{split_name}"):
        try:
            # Download image
            url = row['image_link']
            response = requests.get(url, timeout=10)
            img = Image.open(BytesIO(response.content)).convert("RGB")
            
            # Process image
            inputs = processor(images=[img], return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Extract features
            with torch.no_grad():
                features = model.get_image_features(**inputs)
            
            embeddings.append(features.cpu().numpy()[0])
            
        except Exception as e:
            # Failed - use zero vector
            embeddings.append(np.zeros(1024))
            failed_indices.append(idx)
            
            if len(failed_indices) <= 10:  # Log first 10 failures
                print(f"  Failed idx {idx}: {str(e)[:50]}")
    
    embeddings_array = np.array(embeddings)
    
    print(f"\n✓ Extracted {len(embeddings)} embeddings")
    print(f"✓ Shape: {embeddings_array.shape}")
    print(f"✓ Failed: {len(failed_indices)} ({len(failed_indices)/len(df)*100:.2f}%)")
    print(f"✓ Mean: {embeddings_array.mean():.4f}")
    print(f"✓ Std: {embeddings_array.std():.4f}")
    
    return embeddings_array, failed_indices

# Extract train embeddings
train_start = time.time()
train_embeddings, train_failed = extract_embeddings(df_train, "train")
train_time = time.time() - train_start

# Save train embeddings
np.save('/kaggle/working/train_marqo_embeddings.npy', train_embeddings)
print(f"✓ Saved to train_marqo_embeddings.npy")
print(f"✓ Train extraction time: {train_time:.1f}s ({train_time/len(df_train):.2f}s per image)")

# Extract test embeddings
test_start = time.time()
test_embeddings, test_failed = extract_embeddings(df_test, "test")
test_time = time.time() - test_start

# Save test embeddings
np.save('/kaggle/working/test_marqo_embeddings.npy', test_embeddings)
print(f"✓ Saved to test_marqo_embeddings.npy")
print(f"✓ Test extraction time: {test_time:.1f}s ({test_time/len(df_test):.2f}s per image)")

# Phase 5: Analyze Correlations
print("\n" + "="*80)
print("PHASE 5: ANALYZE CORRELATIONS WITH PRICE")
print("="*80)

try:
    prices = np.log1p(df_train['PRODUCT_PRICE'].values)
    
    print("\nComputing correlations for all 1024 dimensions...")
    correlations = []
    for dim in range(train_embeddings.shape[1]):
        corr, pval = pearsonr(train_embeddings[:, dim], prices)
        correlations.append((dim, abs(corr), pval))
    
    # Sort by absolute correlation
    correlations.sort(key=lambda x: x[1], reverse=True)
    
    # Statistics
    corr_values = [c[1] for c in correlations]
    max_corr = max(corr_values)
    mean_corr = np.mean(corr_values)
    median_corr = np.median(corr_values)
    top10_mean = np.mean([c[1] for c in correlations[:10]])
    
    print(f"\n📊 CORRELATION STATISTICS:")
    print(f"  Max correlation:     {max_corr:.4f}")
    print(f"  Top 10 mean:         {top10_mean:.4f}")
    print(f"  Mean correlation:    {mean_corr:.4f}")
    print(f"  Median correlation:  {median_corr:.4f}")
    
    print(f"\n🔝 TOP 10 DIMENSIONS:")
    for i, (dim, corr, pval) in enumerate(correlations[:10], 1):
        print(f"  {i}. Dim {dim:4d}: corr={corr:.4f}, p={pval:.4e}")
    
    print(f"\n📈 COMPARISON TO BASELINES:")
    print(f"  CLIP max correlation:  0.0089")
    print(f"  Text max correlation:  0.1089")
    print(f"  Marqo max:            {max_corr:.4f}")
    print(f"  ")
    
    if max_corr > 0.05:
        improvement = (max_corr / 0.0089 - 1) * 100
        print(f"  ✅ MARQO IS {improvement:.1f}% BETTER THAN CLIP!")
    elif max_corr > 0.02:
        print(f"  ⚠️  Moderate improvement over CLIP")
    else:
        print(f"  ❌ Still weak correlation (may not help much)")
    
except Exception as e:
    print(f"✗ Correlation analysis failed: {e}")
    traceback.print_exc()

# Phase 6: Final Summary
print("\n" + "="*80)
print("PHASE 6: FINAL SUMMARY")
print("="*80)

total_time = time.time() - start_time

print(f"\n📊 EXECUTION SUMMARY:")
print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
print(f"  Device: {device}")
print(f"  Train samples: {len(df_train):,}")
print(f"  Test samples: {len(df_test):,}")
print(f"  Embedding dim: 1024")

print(f"\n📈 KEY METRICS:")
print(f"\n📈 KEY METRICS:")
print(f"  Max correlation: {max_corr:.4f}")
print(f"  CLIP baseline: 0.0089")
print(f"  Improvement: {(max_corr / 0.0089):.1f}x")

print(f"\n📁 OUTPUT FILES:")
print(f"  ✓ train_marqo_embeddings.npy ({train_embeddings.nbytes / 1024 / 1024:.1f} MB)")
print(f"  ✓ test_marqo_embeddings.npy ({test_embeddings.nbytes / 1024 / 1024:.1f} MB)")
print(f"  ✓ KAGGLE_RUN_RESULTS.md (this file)")

print(f"\n🎯 VERDICT:")
if max_corr > 0.05:
    print(f"  ✅ SUCCESS! Use these embeddings in training.")
    print(f"  Expected improvement: 5-8% SMAPE reduction")
    print(f"  Target: 50-52% SMAPE (from 57.75%)")
elif max_corr > 0.02:
    print(f"  ⚠️  MODERATE. Worth testing in model.")
    print(f"  Expected improvement: 2-4% SMAPE reduction")
else:
    print(f"  ❌ WEAK SIGNAL. Try alternative approaches.")
    print(f"  Consider: aesthetic scorer or image metadata")

print(f"\n✅ SCRIPT COMPLETED at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("```")
print()
print("---")
print()
print("## Next Steps")
print()
if max_corr > 0.02:
    print("1. Download `train_marqo_embeddings.npy` and `test_marqo_embeddings.npy`")
    print("2. Push this file to GitHub: `git add KAGGLE_RUN_RESULTS.md && git commit -m 'Marqo results' && git push`")
    print("3. Wait for Phase 2 training script")
else:
    print("1. Push this file to GitHub: `git add KAGGLE_RUN_RESULTS.md && git commit -m 'Marqo results' && git push`")
    print("2. Discuss alternative image feature strategies")

# Close tee output
sys.stdout = original_stdout
sys.stderr = original_stderr
results_file.close()

print(f"\n✅ Results saved to /kaggle/working/KAGGLE_RUN_RESULTS.md")
print(f"Download this file and push to GitHub!")
