#!/usr/bin/env python3
"""
STL-10 Dataset Download Script
============================

Downloads and prepares the STL-10 dataset for the hackathon project.
"""

import os
import sys
import urllib.request
import tarfile
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
import argparse
import logging

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.data_loader import STL10DataLoader

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Dataset URLs and info
STL10_URL = "http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz"
STL10_FILENAME = "stl10_binary.tar.gz"
STL10_FOLDER = "stl10_binary"

# Class names for STL-10
CLASS_NAMES = [
    'airplane', 'bird', 'car', 'cat', 'deer',
    'dog', 'horse', 'monkey', 'ship', 'truck'
]


def download_file(url: str, filename: str) -> None:
    """
    Download a file from URL with progress bar.
    
    Parameters:
    -----------
    url : str
        URL to download from
    filename : str
        Local filename to save to
    """
    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, downloaded * 100.0 / total_size)
            sys.stdout.write(f"\rDownloading {filename}: {percent:.1f}% "
                           f"({downloaded // (1024*1024)} MB / {total_size // (1024*1024)} MB)")
            sys.stdout.flush()
    
    logger.info(f"Downloading {filename} from {url}")
    urllib.request.urlretrieve(url, filename, progress_hook)
    print()  # New line after progress
    logger.info(f"Download completed: {filename}")


def extract_archive(archive_path: str, extract_to: str) -> None:
    """
    Extract tar.gz archive.
    
    Parameters:
    -----------
    archive_path : str
        Path to archive file
    extract_to : str
        Directory to extract to
    """
    logger.info(f"Extracting {archive_path} to {extract_to}")
    
    with tarfile.open(archive_path, 'r:gz') as tar:
        tar.extractall(extract_to)
    
    logger.info("Extraction completed")


def verify_dataset(data_dir: str) -> bool:
    """
    Verify that dataset was downloaded and extracted correctly.
    
    Parameters:
    -----------
    data_dir : str
        Path to dataset directory
        
    Returns:
    --------
    bool
        True if dataset is valid
    """
    expected_files = [
        'train_X.bin', 'train_y.bin',
        'test_X.bin', 'test_y.bin',
        'unlabeled_X.bin',
        'fold_indices.txt', 'class_names.txt'
    ]
    
    stl10_dir = os.path.join(data_dir, STL10_FOLDER)
    
    if not os.path.exists(stl10_dir):
        logger.error(f"STL-10 directory not found: {stl10_dir}")
        return False
    
    missing_files = []
    for filename in expected_files:
        filepath = os.path.join(stl10_dir, filename)
        if not os.path.exists(filepath):
            missing_files.append(filename)
    
    if missing_files:
        logger.error(f"Missing files: {missing_files}")
        return False
    
    # Check file sizes (rough validation)
    file_sizes = {
        'train_X.bin': 150000000,  # ~150MB
        'test_X.bin': 120000000,   # ~120MB  
        'unlabeled_X.bin': 1200000000,  # ~1.2GB
    }
    
    for filename, expected_size in file_sizes.items():
        filepath = os.path.join(stl10_dir, filename)
        actual_size = os.path.getsize(filepath)
        if abs(actual_size - expected_size) > expected_size * 0.1:  # 10% tolerance
            logger.warning(f"Unexpected size for {filename}: {actual_size} bytes "
                          f"(expected ~{expected_size} bytes)")
    
    logger.info("Dataset verification passed")
    return True


def create_class_names_file(data_dir: str) -> None:
    """
    Create class_names.txt file if it doesn't exist.
    
    Parameters:
    -----------
    data_dir : str
        Path to dataset directory
    """
    class_names_path = os.path.join(data_dir, STL10_FOLDER, 'class_names.txt')
    
    if not os.path.exists(class_names_path):
        logger.info("Creating class_names.txt file")
        with open(class_names_path, 'w') as f:
            for i, name in enumerate(CLASS_NAMES):
                f.write(f"{i+1} {name}\n")


def test_data_loading(data_dir: str) -> None:
    """
    Test that data can be loaded correctly.
    
    Parameters:
    -----------
    data_dir : str
        Path to dataset directory
    """
    logger.info("Testing data loading...")
    
    try:
        # Initialize dataset
        dataset = STL10DataLoader(data_dir)
        
        # Load a small sample
        logger.info("Loading training and test data...")
        X_train, y_train, X_test, y_test = dataset.load_train_test()
        logger.info(f"Training data count: {len(X_train)}, labels count: {len(y_train)}")
        logger.info(f"Test data count: {len(X_test)}, labels count: {len(y_test)}")
        
        logger.info("Loading unlabeled data sample...")
        X_unlabeled = dataset.load_unlabeled()[:1000]  # Solo primeros 1000
        logger.info(f"Unlabeled data count: {len(X_unlabeled)}")
        
        # Verify data properties
        # Convert first image to numpy for verification
        sample_img = np.array(X_train[0])
        assert sample_img.shape == (96, 96, 3), f"Unexpected image shape: {sample_img.shape}"
        
        # Accept both uint8 (0-255) and float32 (0.0-1.0) formats
        assert sample_img.dtype in [np.uint8, np.float32], f"Unexpected dtype: {sample_img.dtype}"
        
        # Verify value ranges based on dtype
        if sample_img.dtype == np.uint8:
            assert 0 <= sample_img.min() and sample_img.max() <= 255, f"Invalid value range for uint8: [{sample_img.min()}, {sample_img.max()}]"
        elif sample_img.dtype == np.float32:
            assert 0.0 <= sample_img.min() and sample_img.max() <= 1.0, f"Invalid value range for float32: [{sample_img.min()}, {sample_img.max()}]"
        
        assert len(np.unique(y_train)) == 10, f"Unexpected number of classes: {len(np.unique(y_train))}"
        
        logger.info("Data loading test passed!")
        
        # Print some statistics
        print("\nDataset Statistics:")
        print("=" * 40)
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Image dimensions: {np.array(X_train[0]).shape}")
        print(f"Data type: {sample_img.dtype}")
        print(f"Value range: [{sample_img.min()}, {sample_img.max()}]")
        print(f"Classes: {sorted(np.unique(y_train))}")
        
        # Class distribution
        unique, counts = np.unique(y_train, return_counts=True)
        print(f"\nTraining class distribution:")
        for cls, count in zip(unique, counts):
            class_name = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else f"Class {cls}"
            print(f"  {class_name}: {count} samples")
        
    except Exception as e:
        logger.error(f"Data loading test failed: {e}")
        raise


def main():
    """Main download and setup function."""
    parser = argparse.ArgumentParser(description='Download and setup STL-10 dataset')
    parser.add_argument('--data-dir', type=str, default='data/',
                       help='Directory to download data to (default: data/)')
    parser.add_argument('--force-download', action='store_true',
                       help='Force re-download even if data exists')
    parser.add_argument('--test-only', action='store_true',
                       help='Only test data loading (skip download)')
    parser.add_argument('--extract-only', action='store_true',
                       help='Only extract existing archive (skip download)')
    
    args = parser.parse_args()
    
    # Create data directory
    data_dir = Path(args.data_dir)
    data_dir.mkdir(exist_ok=True)

    archive_path = data_dir / "raw" / STL10_FILENAME
    stl10_dir = data_dir / "raw" / STL10_FOLDER

    # Check if we need to download
    need_download = True
    if args.test_only:
        need_download = False
    elif args.extract_only:
        need_download = False
        if not archive_path.exists():
            logger.error(f"Archive not found for extraction: {archive_path}")
            return 1
    elif stl10_dir.exists() and not args.force_download:
        logger.info("STL-10 directory already exists. Use --force-download to re-download.")
        need_download = False
    elif archive_path.exists() and not args.force_download:
        logger.info("Archive already downloaded. Skipping download.")
        need_download = False
    
    try:
        # Download if needed
        if need_download and not args.test_only and not args.extract_only:
            download_file(STL10_URL, str(archive_path))
        
        # Extract if needed
        if not args.test_only and (need_download or args.extract_only or not stl10_dir.exists()):
            if archive_path.exists():
                extract_archive(str(archive_path), str(data_dir))
                
                # Clean up archive unless specifically keeping it
                if not args.force_download:
                    logger.info("Removing archive file to save space...")
                    archive_path.unlink()
            else:
                logger.error(f"Archive file not found: {archive_path}")
                return 1
        
        # Create class names file
        if stl10_dir.exists():
            create_class_names_file(str(data_dir))
        
        # Verify dataset
        if stl10_dir.exists():
            if not verify_dataset(str(data_dir)):
                logger.error("Dataset verification failed!")
                return 1
        
        # Test data loading
        if stl10_dir.exists():
            test_data_loading(str(data_dir))
        
        print("\n" + "="*50)
        print("STL-10 Dataset Setup Complete!")
        print("="*50)
        print(f"Dataset location: {stl10_dir.absolute()}")
        print("You can now run the training and evaluation scripts.")
        print("\nNext steps:")
        print("  1. Run training: python scripts/train_descriptors.py")
        print("  2. Run evaluation: python scripts/evaluate_descriptors.py")
        print("  3. Check results in: results/")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Download interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Error during setup: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())