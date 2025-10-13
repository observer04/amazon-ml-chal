#!/bin/bash
# Kaggle installation script for Amazon ML Challenge
# Run this first on Kaggle to install all required packages

echo "=================================================================================="
echo "KAGGLE INSTALLATION: Amazon ML Challenge Dependencies"
echo "=================================================================================="
echo

echo "📦 Installing core packages..."
pip install --upgrade pip --quiet

echo "📦 Installing PyTorch ecosystem..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --quiet

echo "📦 Installing OpenCLIP (for Marqo-ecommerce model)..."
pip install open_clip_torch --quiet

echo "📦 Installing transformers and other ML packages..."
pip install transformers sentence-transformers --quiet

echo "📦 Installing data science packages..."
pip install numpy pandas scikit-learn scipy --quiet

echo "📦 Installing image processing..."
pip install Pillow requests --quiet

echo "📦 Installing utilities..."
pip install tqdm matplotlib seaborn lightgbm --quiet

echo "📦 Installing Jupyter (optional)..."
pip install jupyter ipykernel --quiet

echo
echo "=================================================================================="
echo "✅ INSTALLATION COMPLETE"
echo "=================================================================================="
echo
echo "Now you can run:"
echo "  ./kaggle_auto_switch.sh"
echo
echo "Or run individual scripts:"
echo "  python kaggle_marqo_openclip_fixed.py"
echo "  python kaggle_extract_marqo_embeddings.py"
echo "  python kaggle_plan_b_fashion_siglip.py"
echo "  python kaggle_plan_c_google_siglip.py"
echo "=================================================================================="