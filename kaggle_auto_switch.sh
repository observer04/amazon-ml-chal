#!/bin/bash
# Auto-switch script: Try Marqo-ecommerce, fallback to FashionSigLIP

echo "=================================================================================="
echo "AUTO-SWITCH: Marqo-ecommerce → FashionSigLIP fallback"
echo "=================================================================================="
echo

echo "🔄 STEP 1: Testing Marqo-ecommerce model..."
echo "Command: python kaggle_extract_marqo_embeddings.py"
echo

# Try Marqo-ecommerce first
if python kaggle_extract_marqo_embeddings.py; then
    echo
    echo "✅ SUCCESS: Marqo-ecommerce worked!"
    echo "Embeddings saved as train_marqo_embeddings.npy and test_marqo_embeddings.npy"
    exit 0
else
    echo
    echo "❌ FAILED: Marqo-ecommerce has meta tensor issues"
    echo "🔄 STEP 2: Switching to FashionSigLIP (Plan B)..."
    echo "Command: python kaggle_plan_b_fashion_siglip.py"
    echo

    # Try FashionSigLIP
    if python kaggle_plan_b_fashion_siglip.py; then
        echo
        echo "✅ SUCCESS: FashionSigLIP worked!"
        echo "Embeddings saved as outputs/train_fashion_siglip_embeddings.npy"
        echo "and outputs/test_fashion_siglip_embeddings.npy"
        exit 0
    else
        echo
        echo "❌ FAILED: Both models failed!"
        echo "🔍 TROUBLESHOOTING:"
        echo "1. Check GPU memory: nvidia-smi"
        echo "2. Check internet: ping -c 3 huggingface.co"
        echo "3. Check disk space: df -h"
        echo "4. Try CPU-only: export CUDA_VISIBLE_DEVICES=''"
        exit 1
    fi
fi