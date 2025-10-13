#!/usr/bin/env python3
"""
KAGGLE PHASE 1: Extract Marqo E-commerce Image Embeddings
All output to stdout/stderr (console)
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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONWARNINGS'] = 'ignore'

print("="*80)
print("KAGGLE: Marqo E-commerce Image Embedding Extraction")
print("="*80)

start_time = time.time()

try:
    # PHASE 1: Install
    print("\n" + "="*80)
    print("PHASE 1: INSTALL PACKAGES")
    print("="*80)
    
    packages = ['transformers', 'torch', 'torchvision', 'pillow', 'accelerate', 
                'sentencepiece', 'protobuf', 'scipy', 'ftfy', 'open_clip_torch']
    
    for pkg in packages:
        print(f"Installing {pkg}...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", pkg],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    print("✓ All packages installed\n")
    
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
    print("="*80)
    print("PHASE 2: LOAD DATA")
    print("="*80)
    
    try:
        df_train = pd.read_csv('/kaggle/working/amazon-ml-chal/dataset/train.csv')
        df_test = pd.read_csv('/kaggle/working/amazon-ml-chal/dataset/test.csv')
        print(f"✓ Loaded from /kaggle/working/amazon-ml-chal/dataset/")
    except:
        df_train = pd.read_csv('dataset/train.csv')
        df_test = pd.read_csv('dataset/test.csv')
        print(f"✓ Loaded from dataset/")
    
    print(f"✓ Train: {len(df_train):,} samples")
    print(f"✓ Test: {len(df_test):,} samples\n")
    
    # PHASE 3: Load Model
    print("="*80)
    print("PHASE 3: LOAD MARQO MODEL")
    print("="*80)
    
    model_start = time.time()
    model_name = 'Marqo/marqo-ecommerce-embeddings-L'
    print(f"Loading {model_name} (652M params)...")
    
    # Load with FP16 to avoid meta tensor issues
    model = AutoModel.from_pretrained(
        model_name, 
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Moving model to {device}...")
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model loaded on {device}")
    print(f"✓ Load time: {time.time() - model_start:.1f}s\n")
    
    # PHASE 4: Extract Embeddings
    print("="*80)
    print("PHASE 4: EXTRACT IMAGE EMBEDDINGS")
    print("="*80)
    
    def extract_embeddings(df, split_name):
        embeddings = []
        failed = []
        
        print(f"\nProcessing {len(df):,} {split_name} images...")
        
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
                    print(f"  Failed idx {idx}: {str(e)[:50]}", file=sys.stderr)
            
            if (idx + 1) % 1000 == 0:
                print(f"  Processed {idx+1:,}/{len(df):,}...")
        
        arr = np.array(embeddings)
        print(f"✓ Shape: {arr.shape}")
        print(f"✓ Failed: {len(failed)} ({len(failed)/len(df)*100:.2f}%)")
        print(f"✓ Mean: {arr.mean():.4f}, Std: {arr.std():.4f}")
        
        return arr, failed
    
    # Train
    train_start = time.time()
    train_embeddings, train_failed = extract_embeddings(df_train, "train")
    train_time = time.time() - train_start
    
    np.save('train_marqo_embeddings.npy', train_embeddings)
    print(f"✓ Saved train_marqo_embeddings.npy")
    print(f"✓ Time: {train_time:.1f}s ({train_time/len(df_train):.3f}s per image)\n")
    
    # Test
    test_start = time.time()
    test_embeddings, test_failed = extract_embeddings(df_test, "test")
    test_time = time.time() - test_start
    
    np.save('test_marqo_embeddings.npy', test_embeddings)
    print(f"✓ Saved test_marqo_embeddings.npy")
    print(f"✓ Time: {test_time:.1f}s ({test_time/len(df_test):.3f}s per image)\n")
    
    # PHASE 5: Analyze Correlations
    print("="*80)
    print("PHASE 5: CORRELATION ANALYSIS")
    print("="*80)
    
    prices = np.log1p(df_train['PRODUCT_PRICE'].values)
    
    print("\nComputing correlations for 1024 dimensions...")
    correlations = []
    for dim in range(1024):
        corr, pval = pearsonr(train_embeddings[:, dim], prices)
        correlations.append((dim, abs(corr), pval))
    
    correlations.sort(key=lambda x: x[1], reverse=True)
    
    max_corr = correlations[0][1]
    mean_corr = np.mean([c[1] for c in correlations])
    top10_mean = np.mean([c[1] for c in correlations[:10]])
    
    print(f"\n📊 STATISTICS:")
    print(f"  Max correlation:  {max_corr:.4f}")
    print(f"  Top 10 mean:      {top10_mean:.4f}")
    print(f"  Mean correlation: {mean_corr:.4f}")
    
    print(f"\n🔝 TOP 10 DIMENSIONS:")
    for i, (dim, corr, pval) in enumerate(correlations[:10], 1):
        print(f"  {i}. Dim {dim:4d}: corr={corr:.4f}, p={pval:.4e}")
    
    print(f"\n📈 COMPARISON:")
    print(f"  CLIP (baseline):  0.0089")
    print(f"  Text (baseline):  0.1089")
    print(f"  Marqo (this run): {max_corr:.4f}")
    print(f"  Improvement:      {(max_corr/0.0089):.1f}x over CLIP")
    
    # VERDICT
    print(f"\n🎯 VERDICT:")
    if max_corr > 0.05:
        print(f"  ✅ SUCCESS! Marqo embeddings are significantly better.")
        print(f"  Expected SMAPE improvement: 5-8%")
        print(f"  Next: Train model with these features")
        verdict = "SUCCESS"
    elif max_corr > 0.02:
        print(f"  ⚠️  MODERATE improvement. Worth testing.")
        print(f"  Expected SMAPE improvement: 2-4%")
        print(f"  Next: Test in ensemble model")
        verdict = "MODERATE"
    else:
        print(f"  ❌ WEAK signal. Consider alternatives.")
        print(f"  Next: Try aesthetic scorer or image metadata")
        verdict = "WEAK"
    
    # Summary
    total_time = time.time() - start_time
    print(f"\n" + "="*80)
    print(f"EXECUTION SUMMARY")
    print(f"="*80)
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Max correlation: {max_corr:.4f}")
    print(f"Verdict: {verdict}")
    print(f"\n✅ COMPLETED at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n" + "="*80)
    print(f"✅ SUCCESS - Max correlation: {max_corr:.4f} (Verdict: {verdict})")
    print("="*80)
    
except Exception as e:
    error_phase = "UNKNOWN"
    if 'install' in str(e).lower():
        error_phase = "INSTALL"
    elif 'csv' in str(e).lower() or 'dataframe' in str(e).lower():
        error_phase = "DATA_LOAD"
    elif 'model' in str(e).lower() or 'marqo' in str(e).lower():
        error_phase = "MODEL_LOAD"
    elif 'extract' in str(e).lower() or 'image' in str(e).lower():
        error_phase = "EXTRACTION"
    else:
        error_phase = "CORRELATION"
    
    print(f"\n" + "="*80, file=sys.stderr)
    print(f"❌ FAILED at phase: {error_phase}", file=sys.stderr)
    print(f"="*80, file=sys.stderr)
    print(f"Error: {e}", file=sys.stderr)
    print(f"\nFull traceback:", file=sys.stderr)
    print(traceback.format_exc(), file=sys.stderr)
    sys.exit(1)
