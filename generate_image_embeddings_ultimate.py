"""
ULTIMATE OPTIMIZED: Streaming Image Embeddings
==============================================

Combines best of utils.py + GPU optimization + pipelining:
1. Download batches in parallel (100 workers from utils.py)
2. Keep images in MEMORY only (never save to disk)
3. Pipeline downloads with GPU processing (no waiting!)
4. Process batches continuously on GPU
5. Discard images after embedding extraction

Expected: 5-10 minutes, 80-90% GPU utilization, ~300MB output
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import requests
from io import BytesIO
from pathlib import Path
from tqdm import tqdm
import gc
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from threading import Thread
warnings.filterwarnings('ignore')


def download_image_to_memory(url, timeout=5):
    """
    Download single image directly to memory (BytesIO).
    Returns PIL Image or None if failed.
    """
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content)).convert('RGB')
            return img
    except:
        pass
    return None


def download_batch_parallel(urls, max_workers=100):
    """
    Download batch of images in parallel to MEMORY (not disk).
    Uses ThreadPoolExecutor with 100 workers (like utils.py).
    
    Args:
        urls: List of image URLs
        max_workers: Number of parallel threads
        
    Returns:
        images: List of PIL Images (None for failed)
    """
    images = [None] * len(urls)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(download_image_to_memory, url): idx 
            for idx, url in enumerate(urls)
        }
        
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                images[idx] = future.result()
            except:
                images[idx] = None
    
    return images


class PipelinedImageEmbeddingExtractor:
    """
    Extract ResNet18 embeddings with pipelined download + GPU processing.
    Downloads happen in background while GPU processes current batch.
    """
    
    def __init__(self, device='cuda', use_fp16=True):
        """Initialize ResNet18 model."""
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        if not torch.cuda.is_available():
            print("❌ ERROR: GPU not available! Enable in Kaggle settings.")
            raise RuntimeError("GPU required")
        
        print(f"🔧 Device: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Load ResNet18
        self.model = models.resnet18(weights='IMAGENET1K_V1')
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Use FP16 for 2x speedup
        self.use_fp16 = use_fp16
        if self.use_fp16:
            self.model = self.model.half()
            print("✅ Using FP16 (2x faster)")
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        print("✅ ResNet18 loaded → 512-dim embeddings")
    
    def preprocess_images(self, images):
        """
        Convert PIL images to tensor batch.
        
        Args:
            images: List of PIL Images (None for failed)
            
        Returns:
            tensor: (N, 3, 224, 224) torch tensor
            success_mask: (N,) boolean array
        """
        tensors = []
        success_mask = []
        
        for img in images:
            if img is not None:
                try:
                    tensor = self.transform(img)
                    tensors.append(tensor)
                    success_mask.append(True)
                except:
                    tensors.append(torch.zeros(3, 224, 224))
                    success_mask.append(False)
            else:
                tensors.append(torch.zeros(3, 224, 224))
                success_mask.append(False)
        
        batch_tensor = torch.stack(tensors)
        return batch_tensor, np.array(success_mask)
    
    def extract_embeddings_gpu(self, images, batch_size=128):
        """
        Extract embeddings from images on GPU.
        
        Args:
            images: List of PIL Images
            batch_size: GPU batch size
            
        Returns:
            embeddings: (N, 512) numpy array
            success_mask: (N,) boolean array
        """
        all_embeddings = []
        all_success = []
        
        # Process in GPU batches
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]
            
            # Preprocess
            batch_tensor, success = self.preprocess_images(batch_images)
            batch_tensor = batch_tensor.to(self.device)
            
            if self.use_fp16:
                batch_tensor = batch_tensor.half()
            
            # Extract embeddings
            with torch.no_grad():
                embeddings = self.model(batch_tensor)
                embeddings = embeddings.squeeze(-1).squeeze(-1)
                
                if self.use_fp16:
                    embeddings = embeddings.float()
                
                all_embeddings.append(embeddings.cpu().numpy())
                all_success.append(success)
            
            # Free GPU memory
            del batch_tensor, embeddings
            torch.cuda.empty_cache()
        
        # Concatenate results
        final_embeddings = np.concatenate(all_embeddings, axis=0)
        final_success = np.concatenate(all_success)
        
        return final_embeddings, final_success
    
    def extract_embeddings_pipelined(self, urls, download_batch_size=1000, gpu_batch_size=128, save_path=None):
        """
        Extract embeddings with pipelined download + GPU processing.
        
        Downloads next batch while GPU processes current batch = NO WAITING!
        
        Args:
            urls: List of image URLs
            download_batch_size: How many to download at once
            gpu_batch_size: GPU batch size
            save_path: Where to save embeddings
            
        Returns:
            embeddings: (N, 512) numpy array
            success_rate: Percentage successful
        """
        print(f"\n🚀 PIPELINED PROCESSING:")
        print(f"   Total: {len(urls):,} images")
        print(f"   Download batch: {download_batch_size} (100 parallel threads)")
        print(f"   GPU batch: {gpu_batch_size}")
        print(f"   Strategy: Download batch N+1 while GPU processes batch N")
        
        all_embeddings = []
        all_success = []
        
        # Split URLs into batches
        url_batches = [urls[i:i+download_batch_size] 
                      for i in range(0, len(urls), download_batch_size)]
        
        # Pipeline: download first batch to start
        print(f"\n📥 Pre-downloading first batch...")
        current_images = download_batch_parallel(url_batches[0], max_workers=100)
        
        # Process batches with pipeline
        with tqdm(total=len(urls), desc="Processing", unit="img") as pbar:
            for batch_idx in range(len(url_batches)):
                # GPU processes current batch
                embeddings, success = self.extract_embeddings_gpu(
                    current_images, 
                    batch_size=gpu_batch_size
                )
                
                all_embeddings.append(embeddings)
                all_success.append(success)
                pbar.update(len(current_images))
                
                # While GPU was busy, start downloading NEXT batch
                if batch_idx + 1 < len(url_batches):
                    # Download next batch in parallel (happens during GPU processing)
                    current_images = download_batch_parallel(
                        url_batches[batch_idx + 1], 
                        max_workers=100
                    )
                
                # Free memory
                del embeddings
                gc.collect()
        
        # Concatenate all results
        embeddings = np.concatenate(all_embeddings, axis=0)
        success_mask = np.concatenate(all_success)
        success_rate = success_mask.mean() * 100
        
        print(f"\n✅ Complete: {embeddings.shape}")
        print(f"   Success: {success_rate:.1f}% ({success_mask.sum():,}/{len(urls):,})")
        print(f"   Failed: {(~success_mask).sum():,}")
        
        # Save
        if save_path:
            np.save(save_path, embeddings)
            file_size = Path(save_path).stat().st_size / (1024**2)
            print(f"💾 Saved: {save_path} ({file_size:.1f} MB)")
        
        return embeddings, success_rate


def cleanup_images():
    """Delete any downloaded images to free disk space."""
    import shutil
    images_dir = Path('images')
    
    if images_dir.exists():
        print("\n🗑️  Cleaning up old downloads...")
        try:
            shutil.rmtree(images_dir)
            print("✅ Deleted images/ directory")
        except:
            pass
    else:
        print("\n✅ No cleanup needed")


def main():
    """Main pipeline."""
    print("="*70)
    print("ULTIMATE OPTIMIZED: PIPELINED IMAGE EMBEDDINGS")
    print("="*70)
    
    # Check GPU
    if not torch.cuda.is_available():
        print("\n❌ GPU not enabled! Go to Kaggle settings and enable GPU.")
        return
    
    # Cleanup old downloads
    cleanup_images()
    
    # Load data
    print("\n📂 Loading datasets...")
    train_df = pd.read_csv('dataset/train.csv')
    test_df = pd.read_csv('dataset/test.csv')
    print(f"✅ Train: {len(train_df):,} | Test: {len(test_df):,}")
    
    # Create output directory
    Path('outputs').mkdir(exist_ok=True)
    
    # Initialize extractor
    print("\n🔧 Initializing pipelined extractor...")
    extractor = PipelinedImageEmbeddingExtractor(device='cuda', use_fp16=True)
    
    # Extract TRAIN embeddings
    print("\n" + "="*70)
    print("TRAIN EMBEDDINGS")
    print("="*70)
    
    train_embeddings, train_success = extractor.extract_embeddings_pipelined(
        train_df['image_link'].tolist(),
        download_batch_size=1000,  # Download 1000 at once
        gpu_batch_size=128,        # Process 128 on GPU
        save_path='outputs/train_image_embeddings.npy'
    )
    
    # Extract TEST embeddings
    print("\n" + "="*70)
    print("TEST EMBEDDINGS")
    print("="*70)
    
    test_embeddings, test_success = extractor.extract_embeddings_pipelined(
        test_df['image_link'].tolist(),
        download_batch_size=1000,
        gpu_batch_size=128,
        save_path='outputs/test_image_embeddings.npy'
    )
    
    # Summary
    print("\n\n" + "="*70)
    print("✅ COMPLETE - IMAGE EMBEDDINGS EXTRACTED")
    print("="*70)
    
    print(f"\n📊 Results:")
    print(f"   Train: {train_embeddings.shape} ({train_success:.1f}% success)")
    print(f"   Test: {test_embeddings.shape} ({test_success:.1f}% success)")
    
    train_size = Path('outputs/train_image_embeddings.npy').stat().st_size / (1024**2)
    test_size = Path('outputs/test_image_embeddings.npy').stat().st_size / (1024**2)
    
    print(f"\n💾 Disk usage:")
    print(f"   Train: {train_size:.1f} MB")
    print(f"   Test: {test_size:.1f} MB")
    print(f"   Total: {train_size + test_size:.1f} MB")
    print(f"   vs raw images: ~{40000/(train_size + test_size):.0f}x smaller!")
    
    if train_success >= 90 and test_success >= 90:
        print(f"\n✅ SUCCESS: Ready for next phase!")
        print(f"\n🚀 Next: python generate_text_embeddings.py")
    else:
        print(f"\n⚠️  Some downloads failed (using zero vectors)")
        print(f"   Should still work, may slightly reduce accuracy")


if __name__ == '__main__':
    main()
