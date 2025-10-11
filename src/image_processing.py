"""
Image processing and feature extraction using pre-trained CNN.
Downloads images and extracts 512-dimensional embeddings from ResNet18.
"""

import pandas as pd
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from typing import List, Optional
import os
from tqdm import tqdm
import sys
sys.path.append('..')
from config.config import IMAGE_SIZE, IMAGE_MEAN, IMAGE_STD, IMAGE_MODEL_NAME, IMAGE_EMBEDDING_DIM


class ImageEmbedder:
    """Extract image embeddings using pre-trained ResNet18."""
    
    def __init__(self, model_name: str = IMAGE_MODEL_NAME, device: Optional[str] = None):
        """
        Initialize the image embedder.
        
        Args:
            model_name: 'resnet18' or 'efficientnet_b0'
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"Loading image model: {model_name} on {device}")
        
        # Load pre-trained model
        if model_name == 'resnet18':
            self.model = models.resnet18(pretrained=True)
            # Remove final classification layer
            self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        elif model_name == 'efficientnet_b0':
            self.model = models.efficientnet_b0(pretrained=True)
            self.model.classifier = torch.nn.Identity()
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        self.model = self.model.to(device)
        self.model.eval()
        self.device = device
        self.embedding_dim = IMAGE_EMBEDDING_DIM
        
        # Define image transformations
        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)
        ])
    
    def load_image_from_url(self, url: str, timeout: int = 5) -> Optional[Image.Image]:
        """
        Download and load image from URL.
        
        Args:
            url: Image URL
            timeout: Request timeout in seconds
            
        Returns:
            PIL Image or None if failed
        """
        if pd.isna(url) or url.strip() == '':
            return None
        
        try:
            response = requests.get(url, timeout=timeout)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content)).convert('RGB')
                return img
        except Exception as e:
            # Silently fail for individual images
            return None
        
        return None
    
    def embed_image(self, img: Image.Image) -> np.ndarray:
        """
        Extract embedding from PIL Image.
        
        Args:
            img: PIL Image
            
        Returns:
            numpy array of shape (512,)
        """
        # Transform and add batch dimension
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            embedding = self.model(img_tensor)
        
        # Flatten and convert to numpy
        embedding = embedding.squeeze().cpu().numpy()
        
        return embedding
    
    def embed_from_url(self, url: str) -> Optional[np.ndarray]:
        """
        Load image from URL and extract embedding.
        
        Args:
            url: Image URL
            
        Returns:
            numpy array or None if failed
        """
        img = self.load_image_from_url(url)
        if img is None:
            return None
        
        return self.embed_image(img)
    
    def embed_batch(self, urls: List[str], show_progress: bool = True) -> np.ndarray:
        """
        Extract embeddings from list of URLs.
        Failed images get zero vectors.
        
        Args:
            urls: List of image URLs
            show_progress: Show progress bar
            
        Returns:
            numpy array of shape (len(urls), 512)
        """
        embeddings = []
        
        iterator = tqdm(urls, desc="Processing images") if show_progress else urls
        
        for url in iterator:
            emb = self.embed_from_url(url)
            if emb is None:
                # Use zero vector for failed images
                emb = np.zeros(self.embedding_dim)
            embeddings.append(emb)
        
        return np.array(embeddings)
    
    def embed_dataframe(self, df: pd.DataFrame, url_column: str = 'image_link') -> np.ndarray:
        """
        Extract embeddings for all images in DataFrame.
        
        Args:
            df: DataFrame with image URL column
            url_column: Name of URL column
            
        Returns:
            numpy array of shape (len(df), 512)
        """
        print(f"Extracting image embeddings for {len(df)} images...")
        urls = df[url_column].tolist()
        return self.embed_batch(urls)


def create_image_embeddings(df: pd.DataFrame,
                           url_column: str = 'image_link',
                           embedder: Optional[ImageEmbedder] = None,
                           save_path: Optional[str] = None) -> np.ndarray:
    """
    Create image embeddings for entire DataFrame.
    
    Args:
        df: DataFrame with image URL column
        url_column: Name of URL column
        embedder: Pre-initialized ImageEmbedder (optional)
        save_path: Path to save embeddings (optional)
        
    Returns:
        numpy array of shape (len(df), 512)
    """
    if embedder is None:
        embedder = ImageEmbedder()
    
    embeddings = embedder.embed_dataframe(df, url_column)
    
    if save_path:
        print(f"Saving embeddings to {save_path}")
        np.save(save_path, embeddings)
    
    print(f"Created embeddings with shape: {embeddings.shape}")
    print(f"Images failed: {np.all(embeddings == 0, axis=1).sum()} / {len(embeddings)}")
    
    return embeddings


if __name__ == '__main__':
    # Test image embedding
    test_urls = [
        "https://m.media-amazon.com/images/I/71uRHvXTI8L.jpg",  # Example product image
        "invalid_url",  # Should fail gracefully
    ]
    
    embedder = ImageEmbedder()
    embeddings = embedder.embed_batch(test_urls, show_progress=False)
    
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Sample embedding (first 10 dims): {embeddings[0][:10]}")
    print(f"Failed embeddings: {np.all(embeddings == 0, axis=1).sum()}")
