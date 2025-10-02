#!/usr/bin/env python3
"""
Training Script for Unsupervised Descriptors
==========================================

Trains all descriptor methods on the STL-10 dataset.
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.data_loader import STL10DataLoader
from descriptors.global_descriptors import (
    HOGDescriptor, LBPDescriptor, 
    ColorHistogramDescriptor, GISTDescriptor
)
from descriptors.local_descriptors import (
    SIFTDescriptor, ORBDescriptor, 
    BRISKDescriptor, SURFDescriptor
)
from descriptors.encoding import (
    BagOfWordsEncoder, VLADEncoder, FisherVectorEncoder
)
from utils.preprocessing import ImagePreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DescriptorTrainer:
    """
    Trainer for all descriptor methods.
    """
    
    def __init__(self, 
                 data_dir: str,
                 results_dir: str = "results",
                 cache_dir: str = "cache"):
        """
        Initialize trainer.
        
        Parameters:
        -----------
        data_dir : str
            Path to STL-10 dataset
        results_dir : str
            Directory to save results
        cache_dir : str
            Directory for caching features
        """
        self.data_dir = data_dir
        self.results_dir = Path(results_dir)
        self.cache_dir = Path(cache_dir)
        
        # Create directories
        self.results_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize dataset
        self.dataset = STL10DataLoader(data_dir)
        self.preprocessor = ImagePreprocessor()
        
        # Initialize descriptors
        self.global_descriptors = {
            'hog': HOGDescriptor(),
            'lbp': LBPDescriptor(),
            'color_histogram': ColorHistogramDescriptor(),
            'gist': GISTDescriptor()
        }
        
        self.local_descriptors = {
            'sift': SIFTDescriptor(),
            'orb': ORBDescriptor(),
            'brisk': BRISKDescriptor()
            # 'surf': SURFDescriptor()  # Requires opencv-contrib-python
        }
        
        # Initialize encoders for local descriptors
        self.encoders = {
            'bow': BagOfWordsEncoder(n_clusters=256),
            'vlad': VLADEncoder(n_clusters=64),
            'fisher': FisherVectorEncoder(n_components=32)
        }
        
        self.training_info = {}
    
    def load_training_data(self, max_samples: int = None) -> tuple:
        """
        Load training data for unsupervised learning.
        
        Parameters:
        -----------
        max_samples : int, optional
            Maximum number of samples to load
            
        Returns:
        --------
        tuple
            (images, labels) for training
        """
        logger.info("Loading training data...")
        
        # Load both labeled and unlabeled data for training descriptors
        X_train, y_train = self.dataset.load_train_data()
        X_unlabeled = self.dataset.load_unlabeled_data(max_samples=max_samples//2 if max_samples else None)
        
        # Combine labeled and unlabeled data for training
        if max_samples:
            if len(X_train) > max_samples//2:
                indices = np.random.choice(len(X_train), max_samples//2, replace=False)
                X_train = X_train[indices]
                y_train = y_train[indices]
        
        # Combine datasets
        X_combined = np.vstack([X_train, X_unlabeled])
        y_combined = np.hstack([y_train, np.full(len(X_unlabeled), -1)])  # -1 for unlabeled
        
        logger.info(f"Training data loaded: {len(X_combined)} images "
                   f"({len(X_train)} labeled, {len(X_unlabeled)} unlabeled)")
        
        return X_combined, y_combined
    
    def train_global_descriptors(self, 
                                images: np.ndarray,
                                descriptor_names: List[str] = None) -> Dict[str, Any]:
        """
        Train global descriptors.
        
        Parameters:
        -----------
        images : np.ndarray
            Training images
        descriptor_names : List[str], optional
            Names of descriptors to train
            
        Returns:
        --------
        Dict
            Training results
        """
        if descriptor_names is None:
            descriptor_names = list(self.global_descriptors.keys())
        
        results = {}
        
        for name in descriptor_names:
            if name not in self.global_descriptors:
                logger.warning(f"Unknown descriptor: {name}")
                continue
            
            logger.info(f"Training global descriptor: {name}")
            descriptor = self.global_descriptors[name]
            
            # Measure training time
            start_time = time.time()
            
            try:
                # Preprocess images
                processed_images = []
                for img in images:
                    processed_img = self.preprocessor.preprocess_image(img)
                    processed_images.append(processed_img)
                processed_images = np.array(processed_images)
                
                # Train descriptor
                descriptor.fit(processed_images)
                
                # Extract features to verify
                features = descriptor.extract(processed_images[:100])  # Test on subset
                
                training_time = time.time() - start_time
                
                # Save descriptor
                descriptor_path = self.cache_dir / f"{name}_descriptor.pkl"
                descriptor.save(str(descriptor_path))
                
                results[name] = {
                    'training_time': training_time,
                    'feature_dim': features.shape[1] if len(features.shape) > 1 else len(features[0]),
                    'n_training_samples': len(images),
                    'descriptor_path': str(descriptor_path),
                    'status': 'success'
                }
                
                logger.info(f"✓ {name} trained successfully in {training_time:.2f}s, "
                           f"feature dim: {results[name]['feature_dim']}")
                
            except Exception as e:
                logger.error(f"✗ Failed to train {name}: {e}")
                results[name] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        return results
    
    def train_local_descriptors(self,
                               images: np.ndarray,
                               descriptor_names: List[str] = None) -> Dict[str, Any]:
        """
        Train local descriptors and encoders.
        
        Parameters:
        -----------
        images : np.ndarray
            Training images
        descriptor_names : List[str], optional
            Names of descriptors to train
            
        Returns:
        --------
        Dict
            Training results
        """
        if descriptor_names is None:
            descriptor_names = list(self.local_descriptors.keys())
        
        results = {}
        
        for name in descriptor_names:
            if name not in self.local_descriptors:
                logger.warning(f"Unknown descriptor: {name}")
                continue
            
            logger.info(f"Training local descriptor: {name}")
            descriptor = self.local_descriptors[name]
            
            try:
                start_time = time.time()
                
                # Preprocess images
                processed_images = []
                for img in images:
                    processed_img = self.preprocessor.preprocess_image(img)
                    processed_images.append(processed_img)
                processed_images = np.array(processed_images)
                
                # Train descriptor (fit method for local descriptors just saves parameters)
                descriptor.fit(processed_images)
                
                # Extract features from subset for training encoders
                logger.info(f"Extracting features for encoder training...")
                all_features = []
                feature_extraction_time = time.time()
                
                # Use subset for encoder training to save time
                subset_size = min(1000, len(processed_images))
                subset_indices = np.random.choice(len(processed_images), subset_size, replace=False)
                
                for i, idx in enumerate(subset_indices):
                    if i % 100 == 0:
                        logger.info(f"Processing image {i+1}/{subset_size}")
                    
                    features = descriptor.extract(processed_images[idx:idx+1])
                    if len(features) > 0 and len(features[0]) > 0:
                        all_features.extend(features[0])  # features[0] contains keypoint descriptors
                
                feature_extraction_time = time.time() - feature_extraction_time
                
                if not all_features:
                    logger.error(f"No features extracted for {name}")
                    results[name] = {
                        'status': 'failed',
                        'error': 'No features extracted'
                    }
                    continue
                
                all_features = np.array(all_features)
                logger.info(f"Extracted {len(all_features)} features for {name}")
                
                # Train encoders
                encoder_results = {}
                for encoder_name, encoder in self.encoders.items():
                    logger.info(f"Training {encoder_name} encoder for {name}...")
                    encoder_start = time.time()
                    
                    try:
                        encoder.fit(all_features)
                        
                        # Test encoding
                        test_features = all_features[:100] if len(all_features) > 100 else all_features
                        encoded = encoder.encode(test_features)
                        
                        encoder_time = time.time() - encoder_start
                        
                        # Save encoder
                        encoder_path = self.cache_dir / f"{name}_{encoder_name}_encoder.pkl"
                        encoder.save(str(encoder_path))
                        
                        encoder_results[encoder_name] = {
                            'training_time': encoder_time,
                            'encoded_dim': len(encoded),
                            'encoder_path': str(encoder_path),
                            'status': 'success'
                        }
                        
                        logger.info(f"✓ {encoder_name} encoder trained in {encoder_time:.2f}s, "
                                   f"encoded dim: {len(encoded)}")
                        
                    except Exception as e:
                        logger.error(f"✗ Failed to train {encoder_name} encoder for {name}: {e}")
                        encoder_results[encoder_name] = {
                            'status': 'failed',
                            'error': str(e)
                        }
                
                training_time = time.time() - start_time
                
                # Save descriptor
                descriptor_path = self.cache_dir / f"{name}_descriptor.pkl"
                descriptor.save(str(descriptor_path))
                
                results[name] = {
                    'training_time': training_time,
                    'feature_extraction_time': feature_extraction_time,
                    'n_features_extracted': len(all_features),
                    'n_training_samples': subset_size,
                    'descriptor_path': str(descriptor_path),
                    'encoders': encoder_results,
                    'status': 'success'
                }
                
                logger.info(f"✓ {name} trained successfully in {training_time:.2f}s")
                
            except Exception as e:
                logger.error(f"✗ Failed to train {name}: {e}")
                results[name] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        return results
    
    def save_training_results(self, results: Dict[str, Any], filename: str = "training_results.json") -> None:
        """
        Save training results to file.
        
        Parameters:
        -----------
        results : Dict
            Training results
        filename : str
            Output filename
        """
        results_path = self.results_dir / filename
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Training results saved to: {results_path}")
    
    def print_training_summary(self, results: Dict[str, Any]) -> None:
        """
        Print summary of training results.
        
        Parameters:
        -----------
        results : Dict
            Training results
        """
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        
        global_results = results.get('global_descriptors', {})
        local_results = results.get('local_descriptors', {})
        
        print(f"\nGlobal Descriptors ({len(global_results)} trained):")
        print("-" * 40)
        for name, result in global_results.items():
            if result.get('status') == 'success':
                time_str = f"{result['training_time']:.2f}s"
                dim_str = f"{result['feature_dim']} dims"
                print(f"  ✓ {name:20} {time_str:>8} {dim_str:>10}")
            else:
                print(f"  ✗ {name:20} {'FAILED':>8}")
        
        print(f"\nLocal Descriptors ({len(local_results)} trained):")
        print("-" * 40)
        for name, result in local_results.items():
            if result.get('status') == 'success':
                time_str = f"{result['training_time']:.2f}s"
                features_str = f"{result['n_features_extracted']} features"
                print(f"  ✓ {name:20} {time_str:>8} {features_str:>15}")
                
                # Show encoder results
                for enc_name, enc_result in result.get('encoders', {}).items():
                    if enc_result.get('status') == 'success':
                        enc_time = f"{enc_result['training_time']:.2f}s"
                        enc_dim = f"{enc_result['encoded_dim']} dims"
                        print(f"    └─ {enc_name:15} {enc_time:>8} {enc_dim:>10}")
                    else:
                        print(f"    └─ {enc_name:15} {'FAILED':>8}")
            else:
                print(f"  ✗ {name:20} {'FAILED':>8}")
        
        # Calculate totals
        total_successful = 0
        total_time = 0
        
        for result in global_results.values():
            if result.get('status') == 'success':
                total_successful += 1
                total_time += result.get('training_time', 0)
        
        for result in local_results.values():
            if result.get('status') == 'success':
                total_successful += 1
                total_time += result.get('training_time', 0)
        
        print(f"\nTotal: {total_successful} descriptors trained in {total_time:.2f}s")
        print(f"Models saved to: {self.cache_dir}")
        print(f"Results saved to: {self.results_dir}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train unsupervised descriptors on STL-10')
    parser.add_argument('--data-dir', type=str, default='data/',
                       help='Directory containing STL-10 dataset (default: data/)')
    parser.add_argument('--results-dir', type=str, default='results/',
                       help='Directory to save results (default: results/)')
    parser.add_argument('--cache-dir', type=str, default='cache/',
                       help='Directory to cache trained models (default: cache/)')
    parser.add_argument('--max-samples', type=int, default=10000,
                       help='Maximum number of training samples (default: 10000)')
    parser.add_argument('--global-only', action='store_true',
                       help='Train only global descriptors')
    parser.add_argument('--local-only', action='store_true',
                       help='Train only local descriptors')
    parser.add_argument('--descriptors', type=str, nargs='+',
                       help='Specific descriptors to train (default: all)')
    
    args = parser.parse_args()
    
    # Validate data directory
    if not os.path.exists(args.data_dir):
        logger.error(f"Data directory not found: {args.data_dir}")
        logger.error("Please run 'python scripts/download_data.py' first")
        return 1
    
    try:
        # Initialize trainer
        trainer = DescriptorTrainer(
            data_dir=args.data_dir,
            results_dir=args.results_dir,
            cache_dir=args.cache_dir
        )
        
        # Load training data
        logger.info(f"Loading training data (max_samples={args.max_samples})...")
        X_train, y_train = trainer.load_training_data(max_samples=args.max_samples)
        
        results = {
            'training_config': {
                'data_dir': args.data_dir,
                'max_samples': args.max_samples,
                'n_training_samples': len(X_train),
                'global_only': args.global_only,
                'local_only': args.local_only,
                'specific_descriptors': args.descriptors
            },
            'global_descriptors': {},
            'local_descriptors': {}
        }
        
        # Train global descriptors
        if not args.local_only:
            logger.info("Training global descriptors...")
            global_descriptors = args.descriptors if args.descriptors else None
            if global_descriptors:
                # Filter to only global descriptors
                available_global = list(trainer.global_descriptors.keys())
                global_descriptors = [d for d in global_descriptors if d in available_global]
            
            global_results = trainer.train_global_descriptors(X_train, global_descriptors)
            results['global_descriptors'] = global_results
        
        # Train local descriptors
        if not args.global_only:
            logger.info("Training local descriptors...")
            local_descriptors = args.descriptors if args.descriptors else None
            if local_descriptors:
                # Filter to only local descriptors
                available_local = list(trainer.local_descriptors.keys())
                local_descriptors = [d for d in local_descriptors if d in available_local]
            
            local_results = trainer.train_local_descriptors(X_train, local_descriptors)
            results['local_descriptors'] = local_results
        
        # Save results
        trainer.save_training_results(results)
        
        # Print summary
        trainer.print_training_summary(results)
        
        print("\n" + "="*60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("Next steps:")
        print("  1. Run evaluation: python scripts/evaluate_descriptors.py")
        print("  2. Check results in: results/")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())