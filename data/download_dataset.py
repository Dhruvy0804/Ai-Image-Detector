"""
Dataset Download and Preparation Script
========================================

Downloads the CIFAKE dataset from Kaggle and prepares it for training.

Prerequisites:
    1. Install kaggle: pip install kaggle
    2. Get your API key from: https://www.kaggle.com/settings
    3. Place kaggle.json in:
       - Windows: C:\\Users\\<username>\\.kaggle\\kaggle.json
       - Linux/Mac: ~/.kaggle/kaggle.json

Usage:
    python download_dataset.py
"""

import os
import sys
import shutil


DATASET_NAME = "birdy654/cifake-real-and-ai-generated-synthetic-images"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))


def check_kaggle():
    """Check if kaggle is installed and configured."""
    try:
        import kaggle
        print("✅ Kaggle package found")
        return True
    except ImportError:
        print("❌ Kaggle package not found. Install with: pip install kaggle")
        return False
    except OSError:
        print("❌ Kaggle API key not found.")
        print("   1. Go to https://www.kaggle.com/settings")
        print("   2. Click 'Create New Token' to download kaggle.json")
        print("   3. Place it in ~/.kaggle/kaggle.json")
        return False


def download_dataset():
    """Download CIFAKE dataset from Kaggle."""
    if not check_kaggle():
        print("\n📋 Alternative: Manual Download")
        print("="*50)
        print("1. Go to: https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images")
        print("2. Click 'Download' button")
        print(f"3. Extract the ZIP to: {DATA_DIR}")
        print("\nExpected structure after extraction:")
        print(f"  {DATA_DIR}/train/REAL/   (50,000 images)")
        print(f"  {DATA_DIR}/train/FAKE/   (50,000 images)")
        print(f"  {DATA_DIR}/test/REAL/    (10,000 images)")
        print(f"  {DATA_DIR}/test/FAKE/    (10,000 images)")
        return False
    
    print(f"\n📥 Downloading {DATASET_NAME}...")
    print(f"   Destination: {DATA_DIR}")
    
    os.system(f'kaggle datasets download -d {DATASET_NAME} -p "{DATA_DIR}" --unzip')
    
    # Verify download
    verify_dataset()
    return True


def verify_dataset():
    """Verify the dataset structure is correct."""
    print("\n🔍 Verifying dataset structure...")
    
    expected_dirs = [
        os.path.join(DATA_DIR, 'train', 'REAL'),
        os.path.join(DATA_DIR, 'train', 'FAKE'),
        os.path.join(DATA_DIR, 'test', 'REAL'),
        os.path.join(DATA_DIR, 'test', 'FAKE'),
    ]
    
    all_good = True
    total_images = 0
    
    for dir_path in expected_dirs:
        if os.path.exists(dir_path):
            count = len([f for f in os.listdir(dir_path) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            total_images += count
            print(f"  ✅ {dir_path}: {count:,} images")
        else:
            print(f"  ❌ Missing: {dir_path}")
            all_good = False
    
    if all_good:
        print(f"\n✅ Dataset verified! Total: {total_images:,} images")
    else:
        print("\n⚠️ Some directories are missing.")
        print("Please download the dataset manually from Kaggle.")
    
    return all_good


def get_dataset_stats():
    """Get basic statistics of the dataset."""
    stats = {}
    
    for split in ['train', 'test']:
        for cls in ['REAL', 'FAKE']:
            dir_path = os.path.join(DATA_DIR, split, cls)
            if os.path.exists(dir_path):
                files = [f for f in os.listdir(dir_path) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                stats[f"{split}/{cls}"] = len(files)
    
    return stats


if __name__ == '__main__':
    print("="*60)
    print("📦 CIFAKE Dataset Downloader")
    print("   Real and AI-Generated Synthetic Images")
    print("="*60)
    
    if len(sys.argv) > 1 and sys.argv[1] == '--verify':
        verify_dataset()
    else:
        download_dataset()
