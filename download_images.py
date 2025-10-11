"""
Phase 2: Image Download
========================

Download all training and test images from Amazon CDN.

Using the provided src/utils.py download_images() function
with 100 worker threads for parallel downloads.

Expected:
- Train: 75,000 images
- Test: 75,000 images  
- Success rate: 95%+
- Time: 1.5-2 hours total
"""

import pandas as pd
import os
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from utils import download_images


def download_phase():
    """Download all images for training and test sets."""
    print("="*60)
    print("PHASE 2: IMAGE DOWNLOAD")
    print("="*60)
    
    # Load datasets
    print("\n📂 Loading datasets...")
    train_df = pd.read_csv('dataset/train.csv')
    test_df = pd.read_csv('dataset/test.csv')
    
    print(f"✅ Train: {len(train_df):,} samples")
    print(f"✅ Test: {len(test_df):,} samples")
    
    # Create image directories
    train_img_dir = Path('images/train')
    test_img_dir = Path('images/test')
    
    train_img_dir.mkdir(parents=True, exist_ok=True)
    test_img_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Image directories:")
    print(f"   Train: {train_img_dir}")
    print(f"   Test: {test_img_dir}")
    
    # Check if already downloaded
    existing_train = len(list(train_img_dir.glob('*.jpg'))) if train_img_dir.exists() else 0
    existing_test = len(list(test_img_dir.glob('*.jpg'))) if test_img_dir.exists() else 0
    
    print(f"\n📊 Current status:")
    print(f"   Train images: {existing_train:,} / {len(train_df):,} ({existing_train/len(train_df)*100:.1f}%)")
    print(f"   Test images: {existing_test:,} / {len(test_df):,} ({existing_test/len(test_df)*100:.1f}%)")
    
    # Download train images
    if existing_train < len(train_df) * 0.95:
        print(f"\n🔽 Downloading TRAIN images...")
        print(f"   Using 100 worker threads (from src/utils.py)")
        print(f"   Estimated time: ~1 hour")
        print(f"   Starting download...")
        
        download_images(train_df['image_link'].tolist(), str(train_img_dir))
        
        # Check results
        downloaded_train = len(list(train_img_dir.glob('*.jpg')))
        success_rate_train = downloaded_train / len(train_df) * 100
        
        print(f"\n✅ Train download complete!")
        print(f"   Downloaded: {downloaded_train:,} / {len(train_df):,}")
        print(f"   Success rate: {success_rate_train:.1f}%")
    else:
        print(f"\n✅ Train images already downloaded ({existing_train:,})")
    
    # Download test images
    if existing_test < len(test_df) * 0.95:
        print(f"\n🔽 Downloading TEST images...")
        print(f"   Using 100 worker threads (from src/utils.py)")
        print(f"   Estimated time: ~1 hour")
        print(f"   Starting download...")
        
        download_images(test_df['image_link'].tolist(), str(test_img_dir))
        
        # Check results
        downloaded_test = len(list(test_img_dir.glob('*.jpg')))
        success_rate_test = downloaded_test / len(test_df) * 100
        
        print(f"\n✅ Test download complete!")
        print(f"   Downloaded: {downloaded_test:,} / {len(test_df):,}")
        print(f"   Success rate: {success_rate_test:.1f}%")
    else:
        print(f"\n✅ Test images already downloaded ({existing_test:,})")
    
    # Final summary
    final_train = len(list(train_img_dir.glob('*.jpg')))
    final_test = len(list(test_img_dir.glob('*.jpg')))
    
    print(f"\n\n{'='*60}")
    print(f"✅ PHASE 2 COMPLETE - IMAGE DOWNLOAD")
    print(f"{'='*60}")
    
    print(f"\n📊 Final Status:")
    print(f"   Train images: {final_train:,} / {len(train_df):,} ({final_train/len(train_df)*100:.1f}%)")
    print(f"   Test images: {final_test:,} / {len(test_df):,} ({final_test/len(test_df)*100:.1f}%)")
    print(f"   Total: {final_train + final_test:,} images downloaded")
    
    # Disk space
    total_size = sum(f.stat().st_size for f in train_img_dir.glob('*.jpg'))
    total_size += sum(f.stat().st_size for f in test_img_dir.glob('*.jpg'))
    total_size_gb = total_size / (1024**3)
    
    print(f"\n💾 Disk usage: {total_size_gb:.2f} GB")
    
    if final_train >= len(train_df) * 0.90 and final_test >= len(test_df) * 0.90:
        print(f"\n✅ SUCCESS: >90% images downloaded for both sets")
        print(f"   Ready for Phase 3 (embedding generation)")
    else:
        print(f"\n⚠️  WARNING: Some images failed to download")
        print(f"   Proceeding anyway - will use zero vectors for missing")
    
    print(f"\n🚀 Next: Phase 3 - Generate image embeddings (ResNet18)")


if __name__ == '__main__':
    download_phase()
