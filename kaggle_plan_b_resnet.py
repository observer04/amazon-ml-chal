#!/usr/bin/env python3
"""
KAGGLE PLAN B: Use ResNet50 for Image Features
Simpler, more reliable than Marqo
"""

import os
import sys
import time
import subprocess
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONWARNINGS'] = 'ignore'

print("="*80)
print("KAGGLE PLAN B: ResNet50 Image Features")
print("="*80)

start_time = time.time()

try:
    # Install
    print("\nInstalling packages...")
    packages = ['torch', 'torchvision', 'pillow', 'scipy', 'numpy', 'pandas']
    for pkg in packages:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", pkg],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    print("✓ Packages installed\n")
    
    import torch
    import torchvision.models as models
    import torchvision.transforms as transforms
    import numpy as np
    import pandas as pd
    from PIL import Image
    import requests
    from io import BytesIO
    from scipy.stats import pearsonr
    
    # Load Data
    print("="*80)
    print("LOAD DATA")
    print("="*80)
    try:
        df_train = pd.read_csv('/kaggle/working/amazon-ml-chal/dataset/train.csv')
        df_test = pd.read_csv('/kaggle/working/amazon-ml-chal/dataset/test.csv')
    except:
        df_train = pd.read_csv('dataset/train.csv')
        df_test = pd.read_csv('dataset/test.csv')
    
    print(f"✓ Train: {len(df_train):,} samples")
    print(f"✓ Test: {len(df_test):,} samples\n")
    
    # Load ResNet50
    print("="*80)
    print("LOAD RESNET50 MODEL")
    print("="*80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Use pretrained ResNet50, remove final classification layer
    model = models.resnet50(pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove FC layer
    model = model.to(device)
    model.eval()
    
    # Image preprocessing
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    print(f"✓ ResNet50 loaded (extracts 2048-dim features)\n")
    
    # Extract Embeddings
    print("="*80)
    print("EXTRACT IMAGE EMBEDDINGS")
    print("="*80)
    
    def extract_embeddings(df, split_name):
        embeddings = []
        failed = []
        
        print(f"\nProcessing {len(df):,} {split_name} images...")
        
        for idx, row in df.iterrows():
            try:
                response = requests.get(row['image_link'], timeout=10)
                img = Image.open(BytesIO(response.content)).convert("RGB")
                
                img_tensor = preprocess(img).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    features = model(img_tensor).squeeze().cpu().numpy()
                
                embeddings.append(features)
            except Exception as e:
                embeddings.append(np.zeros(2048))
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
    
    np.save('train_resnet_embeddings.npy', train_embeddings)
    print(f"✓ Saved train_resnet_embeddings.npy")
    print(f"✓ Time: {train_time:.1f}s ({train_time/len(df_train):.3f}s per image)\n")
    
    # Test
    test_start = time.time()
    test_embeddings, test_failed = extract_embeddings(df_test, "test")
    test_time = time.time() - test_start
    
    np.save('test_resnet_embeddings.npy', test_embeddings)
    print(f"✓ Saved test_resnet_embeddings.npy")
    print(f"✓ Time: {test_time:.1f}s ({test_time/len(df_test):.3f}s per image)\n")
    
    # Analyze Correlations
    print("="*80)
    print("CORRELATION ANALYSIS")
    print("="*80)
    
    prices = np.log1p(df_train['PRODUCT_PRICE'].values)
    
    print("\nComputing correlations for 2048 dimensions...")
    correlations = []
    for dim in range(2048):
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
    print(f"  ResNet (this):    {max_corr:.4f}")
    print(f"  Improvement:      {(max_corr/0.0089):.1f}x over CLIP")
    
    # Verdict
    print(f"\n🎯 VERDICT:")
    if max_corr > 0.05:
        verdict = "SUCCESS"
        print(f"  ✅ SUCCESS! ResNet features work well.")
    elif max_corr > 0.02:
        verdict = "MODERATE"
        print(f"  ⚠️  MODERATE. Worth testing.")
    else:
        verdict = "WEAK"
        print(f"  ❌ WEAK signal.")
    
    # Summary
    total_time = time.time() - start_time
    print(f"\n" + "="*80)
    print(f"EXECUTION SUMMARY")
    print(f"="*80)
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Device: {device}")
    print(f"Max correlation: {max_corr:.4f}")
    print(f"Verdict: {verdict}")
    print(f"\n✅ COMPLETED at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n" + "="*80)
    print(f"✅ SUCCESS - Max correlation: {max_corr:.4f} (Verdict: {verdict})")
    print("="*80)
    
except Exception as e:
    print(f"\n" + "="*80, file=sys.stderr)
    print(f"❌ FAILED", file=sys.stderr)
    print(f"="*80, file=sys.stderr)
    print(f"Error: {e}", file=sys.stderr)
    import traceback
    print(traceback.format_exc(), file=sys.stderr)
    sys.exit(1)
