#!/usr/bin/env python3
"""
KAGGLE PHASE 1: Extract Marqo E-commerce Image Embeddings

All output goes to KAGGLE_RUN_RESULTS.md
Console only shows: ✅ SUCCESS or ❌ FAILED <phase>
"""

import os
import sys
import time
import subprocess
from datetime import datetime
import traceback
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
os.environ['PYTHONWARNINGS'] = 'ignore'

# Open results file
log_file = open('KAGGLE_RUN_RESULTS.md', 'w', buffering=1)

def log(msg):
    log_file.write(msg + '\n')
    log_file.flush()

# Start
log("# KAGGLE RUN RESULTS: Marqo Embedding Extraction")
log(f"**Started:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log("```")

start_time = time.time()

try:
    # PHASE 1: Install
    log("\n" + "="*80)
    log("PHASE 1: INSTALL PACKAGES")
    log("="*80)
    
    packages = ['transformers', 'torch', 'torchvision', 'pillow', 'accelerate', 
                'sentencepiece', 'protobuf', 'scipy', 'ftfy', 'open_clip_torch']
    
    for pkg in packages:
        log(f"Installing {pkg}...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", pkg],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE
        )
    log("✓ All packages installed\n")
    
    # Import
    import torch
    import numpy as np
    import pandas as pd
    from PIL import Image
    import requests
    from io import BytesIO
    from transformers import AutoModel, AutoProcessor
    from scipy.stats import pearsonr
    
    # PHASE 2: Load Data
    log("="*80)
    log("PHASE 2: LOAD DATA")
    log("="*80)
    
    try:
        df_train = pd.read_csv('/kaggle/working/amazon-ml-chal/dataset/train.csv')
        df_test = pd.read_csv('/kaggle/working/amazon-ml-chal/dataset/test.csv')
        log(f"✓ Loaded from /kaggle/working/amazon-ml-chal/dataset/")
    except:
        df_train = pd.read_csv('dataset/train.csv')
        df_test = pd.read_csv('dataset/test.csv')
        log(f"✓ Loaded from dataset/")
    
    log(f"✓ Train: {len(df_train):,} samples")
    log(f"✓ Test: {len(df_test):,} samples\n")
    
    # PHASE 3: Load Model
    log("="*80)
    log("PHASE 3: LOAD MARQO MODEL")
    log("="*80)
    
    model_start = time.time()
    model_name = 'Marqo/marqo-ecommerce-embeddings-L'
    log(f"Loading {model_name} (652M params)...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model directly to device to avoid meta tensor issues
    model = AutoModel.from_pretrained(
        model_name, 
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=None  # Don't use device_map to avoid meta tensors
    )
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    
    # Move to device after loading
    model = model.to(device)
    model.eval()
    
    log(f"✓ Model loaded on {device}")
    log(f"✓ Load time: {time.time() - model_start:.1f}s\n")
    
    # PHASE 4: Extract Embeddings
    log("="*80)
    log("PHASE 4: EXTRACT IMAGE EMBEDDINGS")
    log("="*80)
    
    def extract_embeddings(df, split_name):
        embeddings = []
        failed = []
        
        log(f"\nProcessing {len(df):,} {split_name} images...")
        
        for idx, row in df.iterrows():
            try:
                response = requests.get(row['image_link'], timeout=10)
                img = Image.open(BytesIO(response.content)).convert("RGB")
                
                inputs = processor(images=[img], return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    features = model.get_image_features(**inputs)
                
                embeddings.append(features.cpu().numpy()[0])
            except Exception as e:
                embeddings.append(np.zeros(1024))
                failed.append(idx)
                if len(failed) <= 5:
                    log(f"  Failed idx {idx}: {str(e)[:50]}")
            
            if (idx + 1) % 1000 == 0:
                log(f"  Processed {idx+1:,}/{len(df):,}...")
        
        arr = np.array(embeddings)
        log(f"✓ Shape: {arr.shape}")
        log(f"✓ Failed: {len(failed)} ({len(failed)/len(df)*100:.2f}%)")
        log(f"✓ Mean: {arr.mean():.4f}, Std: {arr.std():.4f}")
        
        return arr, failed
    
    # Train
    train_start = time.time()
    train_embeddings, train_failed = extract_embeddings(df_train, "train")
    train_time = time.time() - train_start
    
    np.save('train_marqo_embeddings.npy', train_embeddings)
    log(f"✓ Saved train_marqo_embeddings.npy")
    log(f"✓ Time: {train_time:.1f}s ({train_time/len(df_train):.3f}s per image)\n")
    
    # Test
    test_start = time.time()
    test_embeddings, test_failed = extract_embeddings(df_test, "test")
    test_time = time.time() - test_start
    
    np.save('test_marqo_embeddings.npy', test_embeddings)
    log(f"✓ Saved test_marqo_embeddings.npy")
    log(f"✓ Time: {test_time:.1f}s ({test_time/len(df_test):.3f}s per image)\n")
    
    # PHASE 5: Analyze Correlations
    log("="*80)
    log("PHASE 5: CORRELATION ANALYSIS")
    log("="*80)
    
    prices = np.log1p(df_train['PRODUCT_PRICE'].values)
    
    log("\nComputing correlations for 1024 dimensions...")
    correlations = []
    for dim in range(1024):
        corr, pval = pearsonr(train_embeddings[:, dim], prices)
        correlations.append((dim, abs(corr), pval))
    
    correlations.sort(key=lambda x: x[1], reverse=True)
    
    max_corr = correlations[0][1]
    mean_corr = np.mean([c[1] for c in correlations])
    top10_mean = np.mean([c[1] for c in correlations[:10]])
    
    log(f"\n📊 STATISTICS:")
    log(f"  Max correlation:  {max_corr:.4f}")
    log(f"  Top 10 mean:      {top10_mean:.4f}")
    log(f"  Mean correlation: {mean_corr:.4f}")
    
    log(f"\n🔝 TOP 10 DIMENSIONS:")
    for i, (dim, corr, pval) in enumerate(correlations[:10], 1):
        log(f"  {i}. Dim {dim:4d}: corr={corr:.4f}, p={pval:.4e}")
    
    log(f"\n📈 COMPARISON:")
    log(f"  CLIP (baseline):  0.0089")
    log(f"  Text (baseline):  0.1089")
    log(f"  Marqo (this run): {max_corr:.4f}")
    log(f"  Improvement:      {(max_corr/0.0089):.1f}x over CLIP")
    
    # VERDICT
    log(f"\n🎯 VERDICT:")
    if max_corr > 0.05:
        log(f"  ✅ SUCCESS! Marqo embeddings are significantly better.")
        log(f"  Expected SMAPE improvement: 5-8%")
        log(f"  Next: Train model with these features")
        verdict = "SUCCESS"
    elif max_corr > 0.02:
        log(f"  ⚠️  MODERATE improvement. Worth testing.")
        log(f"  Expected SMAPE improvement: 2-4%")
        log(f"  Next: Test in ensemble model")
        verdict = "MODERATE"
    else:
        log(f"  ❌ WEAK signal. Consider alternatives.")
        log(f"  Next: Try aesthetic scorer or image metadata")
        verdict = "WEAK"
    
    # Summary
    total_time = time.time() - start_time
    log(f"\n" + "="*80)
    log(f"EXECUTION SUMMARY")
    log(f"="*80)
    log(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    log(f"Device: {device}")
    log(f"PyTorch: {torch.__version__}")
    log(f"Max correlation: {max_corr:.4f}")
    log(f"Verdict: {verdict}")
    log(f"\n✅ COMPLETED at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    log("```")
    log("\n---\n")
    log("## Next Steps")
    log(f"1. Push this file: `git add KAGGLE_RUN_RESULTS.md && git commit -m 'Marqo results' && git push`")
    if max_corr > 0.02:
        log(f"2. Download embeddings: `train_marqo_embeddings.npy`, `test_marqo_embeddings.npy`")
    log(f"3. Wait for Phase 2 training script based on these results")
    
    log_file.close()
    
    print(f"✅ SUCCESS - Max correlation: {max_corr:.4f} (Verdict: {verdict})")
    print(f"   Check KAGGLE_RUN_RESULTS.md for full details")
    
except Exception as e:
    error_phase = "UNKNOWN"
    if 'packages' in str(e).lower() or 'install' in str(e).lower():
        error_phase = "INSTALL"
    elif 'load' in str(e).lower() or 'csv' in str(e).lower():
        error_phase = "DATA_LOAD"
    elif 'model' in str(e).lower() or 'marqo' in str(e).lower():
        error_phase = "MODEL_LOAD"
    elif 'extract' in str(e).lower() or 'embedding' in str(e).lower():
        error_phase = "EXTRACTION"
    else:
        error_phase = "CORRELATION"
    
    log(f"\n❌ FAILED at phase: {error_phase}")
    log(f"Error: {e}")
    log(f"\nFull traceback:")
    log(traceback.format_exc())
    log(f"\n```")
    
    log_file.close()
    
    print(f"❌ FAILED at {error_phase}")
    print(f"   Error: {str(e)[:100]}")
    print(f"   Check KAGGLE_RUN_RESULTS.md for full traceback")
    sys.exit(1)
