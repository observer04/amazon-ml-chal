"""
Feature extraction from catalog_content field.
Handles 3 formats:
1. No bullets (plain text)
2. Bullets only (list of specs)
3. Bullets + description (item name, bullets, detailed description)
"""

import re
import pandas as pd
from typing import Dict, List, Tuple, Optional
import sys
sys.path.append('..')
from config.config import MAX_BULLETS, UNIT_MAPPING, QUALITY_KEYWORDS


def parse_catalog_content(text: str) -> Dict[str, str]:
    """
    Parse catalog_content into structured components.
    
    Returns dict with keys:
    - item_name: Product name (if available)
    - bullets: List of bullet points
    - all_bullets_text: Concatenated bullet text for embeddings
    - description: Detailed description (if available)
    """
    if pd.isna(text) or text.strip() == '':
        return {'item_name': '', 'bullets': [], 'all_bullets_text': '', 'description': ''}
    
    # Check if text contains bullet points
    bullet_pattern = r'(?:^|\n)\s*[-•*]\s*(.+?)(?=\n\s*[-•*]|\n\n|$)'
    bullets = re.findall(bullet_pattern, text, re.MULTILINE | re.DOTALL)
    
    if not bullets:
        # Format 1: No bullets - entire text is description
        return {
            'item_name': '',
            'bullets': [],
            'all_bullets_text': '',
            'description': text.strip()
        }
    
    # Has bullets - extract item name (first non-bullet line if exists)
    lines = text.split('\n')
    item_name = ''
    description = ''
    
    # Find first line before bullets
    bullet_start_idx = -1
    for i, line in enumerate(lines):
        if re.match(r'^\s*[-•*]\s*', line):
            bullet_start_idx = i
            break
    
    if bullet_start_idx > 0:
        # Lines before bullets are item name
        item_name = ' '.join(lines[:bullet_start_idx]).strip()
    
    # Find description after bullets
    bullet_end_idx = -1
    for i in range(len(lines) - 1, -1, -1):
        if re.match(r'^\s*[-•*]\s*', lines[i]):
            bullet_end_idx = i
            break
    
    if bullet_end_idx < len(lines) - 1 and bullet_end_idx >= 0:
        # Lines after bullets are description
        desc_lines = lines[bullet_end_idx + 1:]
        description = ' '.join(desc_lines).strip()
    
    # Clean bullets
    bullets = [b.strip() for b in bullets if b.strip()]
    
    # Concatenate all bullets for text embeddings
    all_bullets_text = '\n'.join(bullets)
    
    return {
        'item_name': item_name,
        'bullets': bullets,
        'all_bullets_text': all_bullets_text,
        'description': description
    }


def extract_value_unit(text: str) -> Tuple[Optional[float], Optional[str]]:
    """
    Extract value and unit from catalog_content.
    Examples:
    - "12 Ounce" -> (12.0, "Ounce")
    - "Pack of 6" -> (6.0, "Count")
    - "16 oz" -> (16.0, "Ounce")
    """
    if pd.isna(text):
        return None, None
    
    # Pattern: number + unit
    # Handle formats: "12 Ounce", "12 ounce", "12oz", "12-oz", "Pack of 6"
    patterns = [
        r'(\d+\.?\d*)\s*(Ounce|ounce|oz|OZ|Oz)',
        r'(\d+\.?\d*)\s*(Fl Oz|fl oz|FL Oz)',
        r'(\d+\.?\d*)\s*(Count|count)',
        r'(\d+\.?\d*)\s*(Pound|pound|lb)',
        r'Pack of (\d+)',
        r'(\d+)\s*Pack',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = float(match.group(1))
            
            # Determine unit
            if 'Pack' in pattern:
                unit = 'Count'
            else:
                unit = match.group(2) if len(match.groups()) > 1 else 'Count'
                # Normalize unit
                unit = UNIT_MAPPING.get(unit, unit)
            
            return value, unit
    
    return None, None


def extract_brand(text: str) -> str:
    """
    Extract brand name from catalog_content.
    Handles format: "Item Name: BRAND PRODUCT DESCRIPTION..."
    """
    if pd.isna(text):
        return ''
    
    # Get first line
    first_line = text.split('\n')[0].strip()
    
    # Strip "Item Name:" prefix if present
    if first_line.startswith('Item Name:'):
        first_line = first_line[10:].strip()  # Remove "Item Name:" (10 chars)
    
    # Extract brand: First capitalized word(s) before comma, dash, or parenthesis
    # Examples:
    #   "Log Cabin Sugar Free Syrup, 24 FL OZ" → "Log Cabin Sugar Free Syrup"
    #   "Vlasic Ovals Hamburger Dill Pickle Chips, Keto Friendly" → "Vlasic Ovals Hamburger Dill Pickle Chips"
    
    # Pattern: Capitalized words until delimiter (comma, dash, parenthesis, or "fl oz"/"ounce" unit indicators)
    match = re.match(r'^([A-Z][a-zA-Z0-9\s&\'.]+?)(?:,|\s-\s|\(|\s+\d+\s*(?:Ounce|ounce|oz|OZ|Fl Oz|fl oz|Count|count|Pound|pound|lb))', first_line)
    
    if match:
        brand = match.group(1).strip()
        
        # Filter out common non-brand words and overly long extractions
        exclude_words = ['Pack', 'Set', 'Bundle', 'Size', 'Box', 'Case', 'Item Name']
        if brand in exclude_words:
            return ''
        
        # If brand is very long (>60 chars), it's likely the full product name
        # Try to extract just first 2-3 words as brand
        if len(brand) > 60:
            words = brand.split()[:3]  # Take first 3 words as likely brand
            brand = ' '.join(words)
        
        return brand
    
    return ''


def extract_quality_indicators(text: str) -> Dict[str, int]:
    """
    Check for quality keywords in catalog_content.
    Returns dict: {keyword_type: 1/0}
    """
    if pd.isna(text):
        return {k: 0 for k in QUALITY_KEYWORDS.keys()}
    
    text_lower = text.lower()
    indicators = {}
    
    for keyword_type, keywords in QUALITY_KEYWORDS.items():
        has_keyword = any(kw.lower() in text_lower for kw in keywords)
        indicators[f'has_{keyword_type}'] = 1 if has_keyword else 0
    
    return indicators


def extract_pack_size(text: str) -> Optional[int]:
    """
    Extract pack size from text.
    Examples: "Pack of 6", "6-Pack", "6 Count"
    """
    if pd.isna(text):
        return None
    
    patterns = [
        r'Pack of (\d+)',
        r'(\d+)[- ]Pack',
        r'(\d+)\s*Count',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return int(match.group(1))
    
    return None


def extract_text_features(text: str) -> Dict[str, float]:
    """
    Extract text statistics: length, word count, etc.
    """
    if pd.isna(text):
        return {
            'text_length': 0,
            'word_count': 0,
            'has_bullets': 0,
            'num_bullets': 0,
        }
    
    # Parse structure
    parsed = parse_catalog_content(text)
    
    return {
        'text_length': len(text),
        'word_count': len(text.split()),
        'has_bullets': 1 if parsed['bullets'] else 0,
        'num_bullets': len(parsed['bullets']),
    }


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create all engineered features from catalog_content.
    
    Args:
        df: DataFrame with 'catalog_content' column
        
    Returns:
        DataFrame with added feature columns
    """
    df = df.copy()
    
    # Extract structured components
    print("Parsing catalog content...")
    parsed = df['catalog_content'].apply(parse_catalog_content)
    df['item_name'] = parsed.apply(lambda x: x['item_name'])
    df['description'] = parsed.apply(lambda x: x['description'])
    df['all_bullets_text'] = parsed.apply(lambda x: x['all_bullets_text'])
    
    # Extract bullets as separate columns
    for i in range(MAX_BULLETS):
        df[f'bullet_{i+1}'] = parsed.apply(lambda x: x['bullets'][i] if i < len(x['bullets']) else '')
    
    # Extract Value and Unit
    print("Extracting value and unit...")
    value_unit = df['catalog_content'].apply(extract_value_unit)
    df['value'] = value_unit.apply(lambda x: x[0])
    df['unit'] = value_unit.apply(lambda x: x[1])
    
    # Extract brand
    print("Extracting brand...")
    df['brand'] = df['catalog_content'].apply(extract_brand)
    
    # Extract quality indicators
    print("Extracting quality indicators...")
    quality = df['catalog_content'].apply(extract_quality_indicators)
    for key in QUALITY_KEYWORDS.keys():
        df[f'has_{key}'] = quality.apply(lambda x: x[f'has_{key}'])
    
    # Extract pack size
    print("Extracting pack size...")
    df['pack_size'] = df['catalog_content'].apply(extract_pack_size)
    
    # Extract size category signals
    print("Extracting size categories...")
    df['is_travel_size'] = df['catalog_content'].str.contains(
        'travel size|sample|trial|mini', case=False, na=False
    ).astype(int)
    
    df['is_bulk'] = df['catalog_content'].str.contains(
        'bulk|wholesale|case of', case=False, na=False
    ).astype(int)
    
    # Extract text features
    print("Extracting text features...")
    text_features = df['catalog_content'].apply(extract_text_features)
    for key in ['text_length', 'word_count', 'has_bullets', 'num_bullets']:
        df[key] = text_features.apply(lambda x: x[key])
    
    # Add description length (just the description part, not entire catalog)
    df['description_length'] = df['description'].fillna('').str.len()
    
    # Calculate price per unit (only for training data with 'price')
    if 'price' in df.columns:
        df['price_per_unit'] = df.apply(
            lambda row: row['price'] / row['value'] if pd.notna(row['value']) and row['value'] > 0 else None,
            axis=1
        )
        
        # Add price segments for SMAPE risk analysis
        df['price_segment'] = pd.cut(
            df['price'],
            bins=[0, 10, 50, 100, float('inf')],
            labels=['Budget', 'Mid-Range', 'Premium', 'Luxury']
        )
    
    print(f"Created {len([col for col in df.columns if col not in ['sample_id', 'catalog_content', 'image_link', 'price']])} new features")
    
    return df


def get_tabular_features(df: pd.DataFrame) -> List[str]:
    """
    Get list of feature columns for LightGBM model.
    Excludes text/image columns and identifiers.
    """
    exclude = ['sample_id', 'catalog_content', 'image_link', 'price', 
               'item_name', 'description'] + [f'bullet_{i}' for i in range(1, MAX_BULLETS+1)]
    
    features = [col for col in df.columns if col not in exclude]
    
    # Only numeric features for LightGBM
    numeric_features = []
    for col in features:
        if df[col].dtype in ['int64', 'float64']:
            numeric_features.append(col)
    
    return numeric_features


if __name__ == '__main__':
    # Test on sample data
    test_text = """Premium Organic Coffee Beans
- 12 Ounce Pack
- 100% Arabica
- Dark Roast
- Fair Trade Certified

Rich and bold flavor profile with notes of chocolate and caramel. Perfect for espresso or drip coffee."""
    
    parsed = parse_catalog_content(test_text)
    print("Parsed structure:")
    print(f"Item name: {parsed['item_name']}")
    print(f"Bullets: {parsed['bullets']}")
    print(f"Description: {parsed['description']}")
    
    value, unit = extract_value_unit(test_text)
    print(f"\nValue: {value}, Unit: {unit}")
    
    brand = extract_brand(test_text)
    print(f"Brand: {brand}")
    
    quality = extract_quality_indicators(test_text)
    print(f"Quality indicators: {quality}")
