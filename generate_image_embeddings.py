"""
Phase 2 (REVISED): Streaming Image Embeddings
==============================================

Instead of downloading all images (40GB), we:
1. Download images in batches (to memory)
2. Extract ResNet18 embeddings immediately
3. Save embeddings to disk (only 300MB total!)
4. Discard images from memory
5. Repeat until done

This solves Kaggle's 20GB output limit.

Expected:
- Train embeddings: 75,000 x 512 dims = ~150MB
- Test embeddings: 75,000 x 512 dims = ~150MB
- Total: 300MB vs 40GB (133x smaller!)
- Time: 1-2 hours with GPU
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
warnings.filterwarnings('ignore')


class ImageEmbeddingExtractor:
    """Extract ResNet18 embeddings from images."""
    
    def __init__(self, device='cuda'):
        """Initialize ResNet18 model (pretrained on ImageNet)."""
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Using device: {self.device}")
        
        # Load pretrained ResNet18
        self.model = models.resnet18(pretrained=True)
        
        # Remove final classification layer to get 512-dim embeddings
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Standard ImageNet preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        print("✅ ResNet18 model loaded (pretrained on ImageNet)")
        print("   Output: 512-dimensional embeddings")
    
    def download_image(self, url, timeout=10):
        """Download single image from URL."""
        try:
            response = requests.get(url, timeout=timeout)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content)).convert('RGB')
                return img
        except Exception as e:
            pass
        return None
    
    def process_batch(self, urls, batch_size=32):
        """
        Download and process batch of images.
        
        Args:
            urls: List of image URLs
            batch_size: Number of images to process at once
            
        Returns:
            embeddings: (N, 512) numpy array
            success_mask: (N,) boolean array indicating successful downloads
        """
        embeddings = []
        success_mask = []
        
        # Process in mini-batches for memory efficiency
        for i in range(0, len(urls), batch_size):
            batch_urls = urls[i:i+batch_size]
            batch_images = []
            batch_success = []
            
            # Download images
            for url in batch_urls:
                img = self.download_image(url)
                if img is not None:
                    try:
                        img_tensor = self.transform(img)
                        batch_images.append(img_tensor)
                        batch_success.append(True)
                    except:
                        batch_success.append(False)
                        batch_images.append(torch.zeros(3, 224, 224))  # Zero tensor for failed
                else:
                    batch_success.append(False)
                    batch_images.append(torch.zeros(3, 224, 224))  # Zero tensor for failed
            
            # Stack into batch tensor
            if batch_images:
                batch_tensor = torch.stack(batch_images).to(self.device)
                
                # Extract embeddings
                with torch.no_grad():
                    batch_embeddings = self.model(batch_tensor)
                    batch_embeddings = batch_embeddings.squeeze(-1).squeeze(-1)  # (N, 512)
                
                embeddings.append(batch_embeddings.cpu().numpy())
                success_mask.extend(batch_success)
                
                # Free GPU memory
                del batch_tensor, batch_embeddings
                torch.cuda.empty_cache()
        
        # Concatenate all batches
        if embeddings:
            all_embeddings = np.concatenate(embeddings, axis=0)
        else:
            all_embeddings = np.zeros((len(urls), 512), dtype=np.float32)
        
        return all_embeddings, np.array(success_mask)
    
    def extract_embeddings(self, urls, batch_size=100, save_path=None):
        """
        Extract embeddings for all URLs with progress bar.
        
        Args:
            urls: List of image URLs
            batch_size: Download batch size
            save_path: Path to save embeddings (optional)
            
        Returns:
            embeddings: (N, 512) numpy array
            success_rate: Float indicating % successful downloads
        """
        print(f"\n🔄 Processing {len(urls):,} images in batches of {batch_size}...")
        
        all_embeddings = []
        all_success = []
        
        # Process in batches with progress bar
        num_batches = (len(urls) + batch_size - 1) // batch_size
        
        for i in tqdm(range(0, len(urls), batch_size), total=num_batches, desc="Extracting embeddings"):
            batch_urls = urls[i:i+batch_size]
            batch_emb, batch_success = self.process_batch(batch_urls, batch_size=32)
            
            all_embeddings.append(batch_emb)
            all_success.append(batch_success)
            
            # Periodically free memory
            if (i // batch_size) % 10 == 0:
                gc.collect()
        
        # Concatenate all results
        embeddings = np.concatenate(all_embeddings, axis=0)
        success_mask = np.concatenate(all_success)
        success_rate = success_mask.mean() * 100
        
        print(f"✅ Embeddings extracted: {embeddings.shape}")
        print(f"   Success rate: {success_rate:.1f}% ({success_mask.sum():,}/{len(urls):,})")
        print(f"   Failed downloads: {(~success_mask).sum():,}")
        
        # Save if path provided
        if save_path:
            np.save(save_path, embeddings)
            print(f"💾 Saved to: {save_path}")
            
            # Print file size
            file_size = Path(save_path).stat().st_size / (1024**2)
            print(f"   File size: {file_size:.1f} MB")
        
        return embeddings, success_rate


def cleanup_images():
    """Delete any existing downloaded images to free disk space."""
    print("\n🗑️  Cleaning up existing image downloads...")
    
    train_img_dir = Path('images/train')
    test_img_dir = Path('images/test')
    
    deleted_count = 0
    freed_space = 0
    
    for img_dir in [train_img_dir, test_img_dir]:
        if img_dir.exists():
            for img_file in img_dir.glob('*.jpg'):
                freed_space += img_file.stat().st_size
                img_file.unlink()
                deleted_count += 1
            
            # Remove empty directory
            try:
                img_dir.rmdir()
            except:
                pass
    
    if deleted_count > 0:
        freed_gb = freed_space / (1024**3)
        print(f"✅ Deleted {deleted_count:,} images")
        print(f"   Freed: {freed_gb:.2f} GB")
    else:
        print("✅ No images to clean up")


def main():
    """Main execution pipeline."""
    print("="*70)
    print("PHASE 2 (REVISED): STREAMING IMAGE EMBEDDINGS")
    print("="*70)
    
    # Step 1: Cleanup existing downloads
    cleanup_images()
    
    # Step 2: Load datasets
    print("\n📂 Loading datasets...")
    train_df = pd.read_csv('dataset/train.csv')
    test_df = pd.read_csv('dataset/test.csv')
    
    print(f"✅ Train: {len(train_df):,} samples")
    print(f"✅ Test: {len(test_df):,} samples")
    
    # Create output directory
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)
    
    # Step 3: Initialize extractor
    print("\n🔧 Initializing ResNet18 embedding extractor...")
    extractor = ImageEmbeddingExtractor(device='cuda')
    
    # Step 4: Extract train embeddings
    print("\n" + "="*70)
    print("EXTRACTING TRAIN EMBEDDINGS")
    print("="*70)
    
    train_embeddings, train_success = extractor.extract_embeddings(
        train_df['image_link'].tolist(),
        batch_size=100,
        save_path='outputs/train_image_embeddings.npy'
    )
    
    # Step 5: Extract test embeddings
    print("\n" + "="*70)
    print("EXTRACTING TEST EMBEDDINGS")
    print("="*70)
    
    test_embeddings, test_success = extractor.extract_embeddings(
        test_df['image_link'].tolist(),
        batch_size=100,
        save_path='outputs/test_image_embeddings.npy'
    )
    
    # Step 6: Summary
    print("\n\n" + "="*70)
    print("✅ PHASE 2 COMPLETE - IMAGE EMBEDDINGS EXTRACTED")
    print("="*70)
    
    print(f"\n📊 Final Status:")
    print(f"   Train embeddings: {train_embeddings.shape} ({train_success:.1f}% success)")
    print(f"   Test embeddings: {test_embeddings.shape} ({test_success:.1f}% success)")
    
    # Calculate total disk usage
    train_size = Path('outputs/train_image_embeddings.npy').stat().st_size / (1024**2)
    test_size = Path('outputs/test_image_embeddings.npy').stat().st_size / (1024**2)
    total_size = train_size + test_size
    
    print(f"\n💾 Disk usage:")
    print(f"   Train embeddings: {train_size:.1f} MB")
    print(f"   Test embeddings: {test_size:.1f} MB")
    print(f"   Total: {total_size:.1f} MB")
    print(f"   Saved: ~{40000/total_size:.0f}x less space than raw images!")
    
    if train_success >= 90 and test_success >= 90:
        print(f"\n✅ SUCCESS: >90% embeddings extracted for both sets")
        print(f"   Ready for Phase 3 (text embeddings + model training)")
    else:
        print(f"\n⚠️  WARNING: Some images failed to download")
        print(f"   Using zero vectors for missing images")
        print(f"   This may slightly reduce model performance")
    
    print(f"\n🚀 Next Steps:")
    print(f"   1. Generate text embeddings (sentence-transformers)")
    print(f"   2. Train multi-modal ensemble")
    print(f"   3. Submit predictions")
    
    print(f"\n💡 No more disk space issues! All raw images processed and discarded.")


if __name__ == '__main__':
    main()
