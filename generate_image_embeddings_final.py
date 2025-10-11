"""
FINAL OPTIMIZED: Use src/utils.py + GPU Batch Processing
=========================================================

Leverages the provided download_images() helper (100 workers)
and adds efficient GPU batch processing.

Strategy:
1. Download images to temp directory (using src/utils.py)
2. Process in large GPU batches
3. Delete images as we go to save space
4. Only keep embeddings

Expected: 5-10 minutes total
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import gc
import sys
import shutil
import warnings
warnings.filterwarnings('ignore')

# Import the provided helper
sys.path.insert(0, str(Path(__file__).parent / 'src'))
from utils import download_images


class ImageEmbeddingExtractor:
    """Extract ResNet18 embeddings using downloaded images."""
    
    def __init__(self, device='cuda', use_fp16=True):
        """Initialize ResNet18 model."""
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Using device: {self.device}")
        
        if not torch.cuda.is_available():
            print("⚠️  WARNING: GPU not available! Enable in Kaggle settings.")
            return
        
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Load pretrained ResNet18
        print("\n📥 Loading ResNet18...")
        self.model = models.resnet18(weights='IMAGENET1K_V1')
        
        # Remove final classification layer to get 512-dim embeddings
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Use FP16 for 2x speedup
        self.use_fp16 = use_fp16 and torch.cuda.is_available()
        if self.use_fp16:
            self.model = self.model.half()
            print("✅ Using FP16 (mixed precision) for 2x speedup")
        
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
        
        print("✅ ResNet18 ready (512-dim embeddings)")
    
    def process_image_folder(self, image_folder, image_links, batch_size=128):
        """
        Process downloaded images from folder.
        
        Args:
            image_folder: Path to folder with downloaded images
            image_links: Original list of URLs (to track order)
            batch_size: GPU batch size
            
        Returns:
            embeddings: (N, 512) numpy array
            success_rate: % successfully processed
        """
        print(f"\n🔄 Processing images from: {image_folder}")
        print(f"   GPU batch size: {batch_size}")
        
        all_embeddings = []
        success_count = 0
        
        # Process in batches
        num_batches = (len(image_links) + batch_size - 1) // batch_size
        
        with tqdm(total=len(image_links), desc="Extracting embeddings", unit="img") as pbar:
            for i in range(0, len(image_links), batch_size):
                batch_links = image_links[i:i+batch_size]
                batch_tensors = []
                batch_success = []
                
                # Load images
                for link in batch_links:
                    filename = Path(link).name
                    img_path = Path(image_folder) / filename
                    
                    if img_path.exists():
                        try:
                            img = Image.open(img_path).convert('RGB')
                            tensor = self.transform(img)
                            batch_tensors.append(tensor)
                            batch_success.append(True)
                            success_count += 1
                        except:
                            batch_tensors.append(torch.zeros(3, 224, 224))
                            batch_success.append(False)
                    else:
                        # Image failed to download
                        batch_tensors.append(torch.zeros(3, 224, 224))
                        batch_success.append(False)
                
                # Stack and process on GPU
                if batch_tensors:
                    batch_tensor = torch.stack(batch_tensors).to(self.device)
                    
                    if self.use_fp16:
                        batch_tensor = batch_tensor.half()
                    
                    # Extract embeddings
                    with torch.no_grad():
                        embeddings = self.model(batch_tensor)
                        embeddings = embeddings.squeeze(-1).squeeze(-1)  # (N, 512)
                        
                        if self.use_fp16:
                            embeddings = embeddings.float()
                        
                        all_embeddings.append(embeddings.cpu().numpy())
                    
                    # Free GPU memory
                    del batch_tensor, embeddings
                    torch.cuda.empty_cache()
                
                pbar.update(len(batch_links))
        
        # Concatenate all batches
        embeddings = np.concatenate(all_embeddings, axis=0)
        success_rate = (success_count / len(image_links)) * 100
        
        print(f"\n✅ Embeddings extracted: {embeddings.shape}")
        print(f"   Success rate: {success_rate:.1f}% ({success_count:,}/{len(image_links):,})")
        
        return embeddings, success_rate


def cleanup_images():
    """Delete any existing downloaded images."""
    import shutil
    print("\n🗑️  Cleaning up existing images...")
    
    images_dir = Path('images')
    if images_dir.exists():
        try:
            shutil.rmtree(images_dir)
            print(f"✅ Deleted images/ directory")
        except Exception as e:
            print(f"⚠️  Error: {e}")
    else:
        print("✅ No images to clean up")


def process_dataset(df, split_name, extractor, temp_folder='temp_images'):
    """
    Download and process images for a dataset split.
    
    Args:
        df: DataFrame with image_link column
        split_name: 'train' or 'test'
        extractor: ImageEmbeddingExtractor instance
        temp_folder: Temporary folder for downloads
        
    Returns:
        embeddings: (N, 512) numpy array
        success_rate: % successfully processed
    """
    print("\n" + "="*70)
    print(f"PROCESSING {split_name.upper()} IMAGES")
    print("="*70)
    
    # Create temp folder
    temp_path = Path(temp_folder)
    temp_path.mkdir(exist_ok=True)
    
    # Step 1: Download images using provided helper (100 workers)
    print(f"\n📥 Downloading {len(df):,} images (100 parallel workers)...")
    image_links = df['image_link'].tolist()
    download_images(image_links, str(temp_path))
    
    # Check download success
    downloaded = len(list(temp_path.glob('*.jpg')))
    download_rate = (downloaded / len(df)) * 100
    print(f"✅ Downloaded: {downloaded:,}/{len(df):,} ({download_rate:.1f}%)")
    
    # Step 2: Process with GPU
    embeddings, success_rate = extractor.process_image_folder(
        temp_path, 
        image_links, 
        batch_size=128
    )
    
    # Step 3: Save embeddings
    output_path = f'outputs/{split_name}_image_embeddings.npy'
    np.save(output_path, embeddings)
    
    file_size = Path(output_path).stat().st_size / (1024**2)
    print(f"💾 Saved: {output_path} ({file_size:.1f} MB)")
    
    # Step 4: Clean up downloaded images
    print(f"\n🗑️  Cleaning up temporary images...")
    shutil.rmtree(temp_path)
    print(f"✅ Deleted {temp_folder}/")
    
    return embeddings, success_rate


def main():
    """Main execution pipeline."""
    print("="*70)
    print("PHASE 2: IMAGE EMBEDDINGS (Using src/utils.py)")
    print("="*70)
    
    # Check GPU
    if not torch.cuda.is_available():
        print("\n❌ ERROR: GPU not available!")
        print("   Go to: Notebook Settings → Accelerator → GPU")
        return
    
    # Cleanup any existing images
    cleanup_images()
    
    # Load datasets
    print("\n📂 Loading datasets...")
    train_df = pd.read_csv('dataset/train.csv')
    test_df = pd.read_csv('dataset/test.csv')
    
    print(f"✅ Train: {len(train_df):,} samples")
    print(f"✅ Test: {len(test_df):,} samples")
    
    # Create output directory
    Path('outputs').mkdir(exist_ok=True)
    
    # Initialize extractor
    print("\n🔧 Initializing ResNet18 extractor...")
    extractor = ImageEmbeddingExtractor(device='cuda', use_fp16=True)
    
    # Process train images
    train_embeddings, train_success = process_dataset(
        train_df, 
        'train', 
        extractor,
        temp_folder='temp_train'
    )
    
    # Process test images
    test_embeddings, test_success = process_dataset(
        test_df, 
        'test', 
        extractor,
        temp_folder='temp_test'
    )
    
    # Final summary
    print("\n\n" + "="*70)
    print("✅ PHASE 2 COMPLETE - IMAGE EMBEDDINGS")
    print("="*70)
    
    print(f"\n📊 Results:")
    print(f"   Train: {train_embeddings.shape} ({train_success:.1f}% success)")
    print(f"   Test: {test_embeddings.shape} ({test_success:.1f}% success)")
    
    train_size = Path('outputs/train_image_embeddings.npy').stat().st_size / (1024**2)
    test_size = Path('outputs/test_image_embeddings.npy').stat().st_size / (1024**2)
    
    print(f"\n💾 Files:")
    print(f"   outputs/train_image_embeddings.npy: {train_size:.1f} MB")
    print(f"   outputs/test_image_embeddings.npy: {test_size:.1f} MB")
    print(f"   Total: {train_size + test_size:.1f} MB")
    
    if train_success >= 90 and test_success >= 90:
        print(f"\n✅ SUCCESS! Ready for Phase 3 (text embeddings)")
    else:
        print(f"\n⚠️  Some images failed (using zero vectors)")
    
    print(f"\n🚀 Next: python generate_text_embeddings.py")


if __name__ == '__main__':
    main()
