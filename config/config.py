"""
Configuration settings for the ML Challenge.
These settings rarely change and are safe to import in Kaggle.
"""

# Paths (adjust for Kaggle environment)
DATA_PATH = '../input/ml-challenge-2025'  # Kaggle input path
OUTPUT_PATH = './'  # Kaggle working directory

# Feature Engineering
MAX_BULLETS = 6  # Maximum number of bullet points to extract
KNOWN_UNITS = ['Ounce', 'ounce', 'oz', 'OZ', 'Oz', 'Count', 'count', 'Fl Oz', 'fl oz', 'FL Oz', 'pound', 'Pound', 'lb']

# Unit normalization mapping
UNIT_MAPPING = {
    'ounce': 'Ounce',
    'oz': 'Ounce',
    'OZ': 'Ounce',
    'Oz': 'Ounce',
    'count': 'Count',
    'fl oz': 'Fl Oz',
    'FL Oz': 'Fl Oz',
    'Pound': 'pound',
    'lb': 'pound',
}

# Quality keywords
QUALITY_KEYWORDS = {
    'premium': ['premium', 'Premium', 'PREMIUM'],
    'organic': ['organic', 'Organic', 'ORGANIC'],
    'gourmet': ['gourmet', 'Gourmet', 'GOURMET'],
    'natural': ['natural', 'Natural', 'NATURAL'],
    'artisan': ['artisan', 'Artisan', 'handmade', 'hand-made'],
    'luxury': ['luxury', 'Luxury', 'deluxe', 'Deluxe'],
}

# Model hyperparameters (will be tuned)
LIGHTGBM_PARAMS = {
    'objective': 'mape',  # Mean Absolute Percentage Error (close to SMAPE)
    'metric': 'mape',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'random_state': 42,
}

# Neural network params
TEXT_EMBEDDING_DIM = 384  # MiniLM output
IMAGE_EMBEDDING_DIM = 512  # ResNet18 output
MLP_HIDDEN_DIMS = [128, 64]
DROPOUT_RATE = 0.3
LEARNING_RATE = 0.001
BATCH_SIZE = 256
EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10

# Cross-validation
N_SPLITS = 5
RANDOM_STATE = 42

# Ensemble weights (to be optimized)
ENSEMBLE_WEIGHTS = {
    'lightgbm': 0.55,
    'text_mlp': 0.25,
    'image_mlp': 0.20,
}

# Image processing
IMAGE_SIZE = (224, 224)
IMAGE_MEAN = [0.485, 0.456, 0.406]  # ImageNet normalization
IMAGE_STD = [0.229, 0.224, 0.225]

# Text embedding model
TEXT_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'

# Image model
IMAGE_MODEL_NAME = 'resnet18'  # Pre-trained on ImageNet
