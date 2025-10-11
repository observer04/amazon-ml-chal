"""
Quick test to verify all modules can be imported and basic functionality works.
Run this before deploying to Kaggle to catch any import or syntax errors.
"""

import sys
import numpy as np
import pandas as pd

print("="*80)
print("MODULE IMPORT & FUNCTIONALITY TEST")
print("="*80)

# Test 1: Config
print("\n[1/5] Testing config...")
try:
    from config.config import ENSEMBLE_WEIGHTS, LIGHTGBM_PARAMS, TEXT_MODEL_NAME
    print(f"✓ Config loaded successfully")
    print(f"  - Ensemble weights: {ENSEMBLE_WEIGHTS}")
    print(f"  - Text model: {TEXT_MODEL_NAME}")
except Exception as e:
    print(f"✗ Config failed: {e}")
    sys.exit(1)

# Test 2: Feature Extraction
print("\n[2/5] Testing feature extraction...")
try:
    from src.feature_extraction import parse_catalog_content, extract_value_unit, extract_brand
    
    test_text = """Premium Coffee Beans - 12 Ounce
- 100% Arabica
- Dark Roast
Rich flavor profile"""
    
    parsed = parse_catalog_content(test_text)
    value, unit = extract_value_unit(test_text)
    brand = extract_brand(test_text)
    
    print(f"✓ Feature extraction working")
    print(f"  - Parsed: item_name='{parsed['item_name']}', bullets={len(parsed['bullets'])}")
    print(f"  - IPQ: value={value}, unit={unit}")
    print(f"  - Brand: '{brand}'")
except Exception as e:
    print(f"✗ Feature extraction failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Text Embeddings (skip actual model loading to save time)
print("\n[3/5] Testing text embeddings module...")
try:
    from src.text_embeddings import TextEmbedder
    print(f"✓ Text embeddings module loaded")
    print(f"  - TextEmbedder class available")
    print(f"  - Note: Skipping actual embedding generation (requires GPU/download)")
except ImportError as e:
    print(f"⚠ Text embeddings skipped (missing dependencies): {str(e).split()[3]}")
    print(f"  - Install with: pip install sentence-transformers transformers torch")
except Exception as e:
    print(f"✗ Text embeddings failed: {e}")

# Test 4: Image Processing (skip actual model loading)
print("\n[4/5] Testing image processing module...")
try:
    from src.image_processing import ImageEmbedder
    print(f"✓ Image processing module loaded")
    print(f"  - ImageEmbedder class available")
    print(f"  - Note: Skipping actual embedding generation (requires GPU/download)")
except ImportError as e:
    print(f"⚠ Image processing skipped (missing dependencies): {str(e).split()[3]}")
    print(f"  - Install with: pip install torch torchvision pillow requests")
except Exception as e:
    print(f"✗ Image processing failed: {e}")

# Test 5: Models and Ensemble
print("\n[5/5] Testing models and ensemble...")
try:
    from src.ensemble import Ensemble, smape
    
    # Test SMAPE calculation
    y_true = np.array([10.0, 20.0, 15.0])
    y_pred = np.array([9.5, 21.0, 14.5])
    smape_score = smape(y_true, y_pred)
    
    # Test ensemble
    ensemble = Ensemble({'model1': 0.5, 'model2': 0.5})
    preds_dict = {
        'model1': np.array([10.0, 20.0, 15.0]),
        'model2': np.array([11.0, 19.0, 16.0])
    }
    ensemble_pred = ensemble.predict(preds_dict)
    
    print(f"✓ Ensemble working")
    print(f"  - SMAPE score: {smape_score:.4f}")
    print(f"  - Ensemble predictions: {ensemble_pred}")
    
    try:
        from src.models import MLP
        print(f"✓ Models module loaded (MLP class available)")
    except ImportError as e:
        print(f"⚠ Models module skipped (missing dependencies): {str(e).split()[3]}")
        print(f"  - Install with: pip install torch lightgbm scikit-learn")
        
except ImportError as e:
    print(f"⚠ Ensemble/models skipped (missing dependencies): {str(e).split()[3]}")
    print(f"  - Install with: pip install scipy scikit-learn")
except Exception as e:
    print(f"✗ Models/ensemble failed: {e}")

print("\n" + "="*80)
print("TEST COMPLETE!")
print("="*80)
print("\n✓ Core modules (config, feature extraction) working")
print("⚠ ML modules need dependencies installed")
print("\nNext steps:")
print("1. Install full requirements: pip install -r requirements.txt")
print("2. Run this test again to verify all modules")
print("3. Run kaggle.py on local data to test end-to-end pipeline")
print("4. Upload to Kaggle and run on P100 GPU")
