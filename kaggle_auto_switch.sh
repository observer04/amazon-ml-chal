#!/bin/bash
# Auto-switch script: Try Marqo-OpenCLIP, fallback to Marqo-transformers, then others

echo "=================================================================================="
echo "AUTO-SWITCH: Marqo-OpenCLIP → Marqo-Transformers → FashionSigLIP → Google SigLIP"
echo "=================================================================================="
echo

echo "🔄 STEP 1: Testing Marqo-ecommerce with OpenCLIP (PROPER WAY)..."
echo "Command: python kaggle_marqo_openclip_fixed.py"
echo

# Try Marqo-ecommerce with OpenCLIP first (the correct way)
if python kaggle_marqo_openclip_fixed.py; then
    echo
    echo "✅ SUCCESS: Marqo-ecommerce with OpenCLIP worked!"
    echo "Embeddings saved as outputs/train_marqo_ecommerce_openclip_embeddings.npy"
    echo "and outputs/test_marqo_ecommerce_openclip_embeddings.npy"
    exit 0
else
    echo
    echo "❌ FAILED: Marqo-OpenCLIP failed"
    echo "🔄 STEP 2: Trying Marqo-ecommerce with transformers..."
    echo "Command: python kaggle_extract_marqo_embeddings.py"
    echo

    # Try Marqo-ecommerce with transformers (fallback)
    if python kaggle_extract_marqo_embeddings.py; then
        echo
        echo "✅ SUCCESS: Marqo-ecommerce (transformers) worked!"
        echo "Embeddings saved as outputs/train_marqo_embeddings.npy"
        echo "and outputs/test_marqo_embeddings.npy"
        exit 0
    else
        echo
        echo "❌ FAILED: Marqo-ecommerce has meta tensor issues"
        echo "🔄 STEP 3: Switching to FashionSigLIP (Plan B)..."
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
            echo "❌ FAILED: FashionSigLIP also failed!"
            echo "🔄 STEP 4: Switching to Google SigLIP (Plan C)..."
            echo "Command: python kaggle_plan_c_google_siglip.py"
            echo

            # Try Google SigLIP
            if python kaggle_plan_c_google_siglip.py; then
                echo
                echo "✅ SUCCESS: Google SigLIP worked!"
                echo "Embeddings saved as outputs/train_google_siglip_embeddings.npy"
                echo "and outputs/test_google_siglip_embeddings.npy"
                exit 0
            else
                echo
                echo "❌ FAILED: All models failed!"
                echo "🔍 TROUBLESHOOTING:"
                echo "1. Check GPU memory: nvidia-smi"
                echo "2. Check internet: ping -c 3 huggingface.co"
                echo "3. Check disk space: df -h"
                echo "4. Try CPU-only: export CUDA_VISIBLE_DEVICES=''"
                echo "5. Check transformers version: pip show transformers"
                echo "6. Try ResNet50 as last resort"
                exit 1
            fi
        fi
    fi
fi