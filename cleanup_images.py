"""
Cleanup Script - Free up disk space on Kaggle
==============================================

Run this FIRST on Kaggle to delete any existing downloaded images
before running the streaming embedding extractor.

This will free up ~19GB of disk space.
"""

from pathlib import Path
import shutil


def cleanup_images():
    """Delete all downloaded images."""
    print("="*60)
    print("CLEANUP: REMOVING DOWNLOADED IMAGES")
    print("="*60)
    
    train_img_dir = Path('images/train')
    test_img_dir = Path('images/test')
    images_root = Path('images')
    
    total_deleted = 0
    total_freed = 0
    
    # Delete train images
    if train_img_dir.exists():
        print(f"\n🗑️  Removing train images...")
        file_count = 0
        size_freed = 0
        
        for img_file in train_img_dir.glob('*.jpg'):
            size_freed += img_file.stat().st_size
            img_file.unlink()
            file_count += 1
        
        total_deleted += file_count
        total_freed += size_freed
        
        print(f"   Deleted: {file_count:,} files")
        print(f"   Freed: {size_freed / (1024**3):.2f} GB")
    
    # Delete test images
    if test_img_dir.exists():
        print(f"\n🗑️  Removing test images...")
        file_count = 0
        size_freed = 0
        
        for img_file in test_img_dir.glob('*.jpg'):
            size_freed += img_file.stat().st_size
            img_file.unlink()
            file_count += 1
        
        total_deleted += file_count
        total_freed += size_freed
        
        print(f"   Deleted: {file_count:,} files")
        print(f"   Freed: {size_freed / (1024**3):.2f} GB")
    
    # Remove empty directories
    if images_root.exists():
        try:
            shutil.rmtree(images_root)
            print(f"\n📁 Removed images/ directory")
        except Exception as e:
            print(f"\n⚠️  Could not remove images/ directory: {e}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"✅ CLEANUP COMPLETE")
    print(f"{'='*60}")
    
    if total_deleted > 0:
        print(f"\n📊 Total:")
        print(f"   Files deleted: {total_deleted:,}")
        print(f"   Space freed: {total_freed / (1024**3):.2f} GB")
        print(f"\n✅ Ready for streaming embedding extraction!")
    else:
        print(f"\n✅ No images to clean up")
        print(f"   You're good to go!")
    
    print(f"\n🚀 Next: Run generate_image_embeddings.py")


if __name__ == '__main__':
    cleanup_images()
