#!/usr/bin/env python3
"""
Evaluation Script for Unsupervised Descriptors
==============================================

Evaluates trained descriptors on the STL-10 dataset.
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
from sklearn.metrics import classification_report

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
from evaluation.metrics import ClassificationMetrics
from evaluation.robustness import RobustnessEvaluator
from evaluation.cross_validation import CrossValidationEvaluator
from evaluation.classifiers import LinearSVMClassifier, RandomForestClassifier, LogisticRegressionClassifier
from utils.preprocessing import ImagePreprocessor, DescriptorPostprocessor
from utils.visualization import ResultsVisualizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Class names for STL-10
CLASS_NAMES = [
    'airplane', 'bird', 'car', 'cat', 'deer',
    'dog', 'horse', 'monkey', 'ship', 'truck'
]


class DescriptorEvaluator:
    """
    Evaluator for all descriptor methods.
    """
    
    def __init__(self, 
                 data_dir: str,
                 cache_dir: str = "cache",
                 results_dir: str = "results"):
        """
        Initialize evaluator.
        
        Parameters:
        -----------
        data_dir : str
            Path to STL-10 dataset
        cache_dir : str
            Directory with cached models
        results_dir : str
            Directory to save results
        """
        self.data_dir = data_dir
        self.cache_dir = Path(cache_dir)
        self.results_dir = Path(results_dir)
        
        # Create results directory
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize dataset and utilities
        self.dataset = STL10DataLoader(data_dir)
        self.preprocessor = ImagePreprocessor()
        self.postprocessor = DescriptorPostprocessor()
        self.metrics = ClassificationMetrics()
        self.visualizer = ResultsVisualizer()
        
        # Initialize evaluators
        self.robustness_evaluator = RobustnessEvaluator()
        self.cv_evaluator = CrossValidationEvaluator()
        
        # Initialize classifiers
        self.classifiers = {
            'linear_svm': LinearSVMClassifier(),
            'random_forest': RandomForestClassifier(),
            'logistic_regression': LogisticRegressionClassifier()
        }
        
        # Descriptor classes mapping
        self.descriptor_classes = {
            'hog': HOGDescriptor,
            'lbp': LBPDescriptor,
            'color_histogram': ColorHistogramDescriptor,
            'gist': GISTDescriptor,
            'sift': SIFTDescriptor,
            'orb': ORBDescriptor,
            'brisk': BRISKDescriptor,
            'surf': SURFDescriptor
        }
        
        # Encoder classes mapping
        self.encoder_classes = {
            'bow': BagOfWordsEncoder,
            'vlad': VLADEncoder,
            'fisher': FisherVectorEncoder
        }
    
    def load_test_data(self) -> tuple:
        """
        Load test data for evaluation.
        
        Returns:
        --------
        tuple
            (X_test, y_test, X_train, y_train)
        """
        logger.info("Loading test data...")
        
        X_train, y_train = self.dataset.load_train_data()
        X_test, y_test = self.dataset.load_test_data()
        
        logger.info(f"Loaded {len(X_train)} training and {len(X_test)} test images")
        
        return X_test, y_test, X_train, y_train
    
    def load_descriptor(self, descriptor_name: str) -> Optional[Any]:
        """
        Load trained descriptor from cache.
        
        Parameters:
        -----------
        descriptor_name : str
            Name of descriptor to load
            
        Returns:
        --------
        descriptor or None
        """
        descriptor_path = self.cache_dir / f"{descriptor_name}_descriptor.pkl"
        
        if not descriptor_path.exists():
            logger.warning(f"Descriptor not found: {descriptor_path}")
            return None
        
        try:
            # Get descriptor class
            if descriptor_name not in self.descriptor_classes:
                logger.error(f"Unknown descriptor class: {descriptor_name}")
                return None
            
            descriptor_class = self.descriptor_classes[descriptor_name]
            descriptor = descriptor_class()
            descriptor.load(str(descriptor_path))
            
            return descriptor
            
        except Exception as e:
            logger.error(f"Failed to load descriptor {descriptor_name}: {e}")
            return None
    
    def load_encoder(self, descriptor_name: str, encoder_name: str) -> Optional[Any]:
        """
        Load trained encoder from cache.
        
        Parameters:
        -----------
        descriptor_name : str
            Name of descriptor
        encoder_name : str
            Name of encoder
            
        Returns:
        --------
        encoder or None
        """
        encoder_path = self.cache_dir / f"{descriptor_name}_{encoder_name}_encoder.pkl"
        
        if not encoder_path.exists():
            logger.warning(f"Encoder not found: {encoder_path}")
            return None
        
        try:
            # Get encoder class
            if encoder_name not in self.encoder_classes:
                logger.error(f"Unknown encoder class: {encoder_name}")
                return None
            
            encoder_class = self.encoder_classes[encoder_name]
            encoder = encoder_class()
            encoder.load(str(encoder_path))
            
            return encoder
            
        except Exception as e:
            logger.error(f"Failed to load encoder {descriptor_name}_{encoder_name}: {e}")
            return None
    
    def extract_features(self, 
                        images: np.ndarray,
                        descriptor_name: str,
                        encoder_name: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Extract features using specified descriptor and encoder.
        
        Parameters:
        -----------
        images : np.ndarray
            Input images
        descriptor_name : str
            Name of descriptor
        encoder_name : str, optional
            Name of encoder (for local descriptors)
            
        Returns:
        --------
        features : np.ndarray or None
        """
        # Load descriptor
        descriptor = self.load_descriptor(descriptor_name)
        if descriptor is None:
            return None
        
        try:
            # Preprocess images
            processed_images = []
            for img in images:
                processed_img = self.preprocessor.preprocess_image(img)
                processed_images.append(processed_img)
            processed_images = np.array(processed_images)
            
            # Check if this is a local descriptor that needs encoding
            is_local = descriptor_name in ['sift', 'orb', 'brisk', 'surf']
            
            if is_local and encoder_name:
                # Load encoder
                encoder = self.load_encoder(descriptor_name, encoder_name)
                if encoder is None:
                    return None
                
                # Extract and encode features
                logger.info(f"Extracting features with {descriptor_name}+{encoder_name}...")
                
                all_features = []
                for i, img in enumerate(processed_images):
                    if i % 100 == 0:
                        logger.info(f"Processing image {i+1}/{len(processed_images)}")
                    
                    # Extract local features
                    local_features = descriptor.extract(img.reshape(1, *img.shape))
                    
                    if len(local_features) > 0 and len(local_features[0]) > 0:
                        # Encode to fixed-length vector
                        encoded_features = encoder.encode(local_features[0])
                        all_features.append(encoded_features)
                    else:
                        # If no features detected, create zero vector
                        dummy_features = np.random.randn(10, encoder.get_feature_dim())
                        encoded_features = encoder.encode(dummy_features)
                        all_features.append(encoded_features)
                
                features = np.array(all_features)
                
            else:
                # Global descriptor - extract directly
                logger.info(f"Extracting features with {descriptor_name}...")
                features = descriptor.extract(processed_images)
            
            # Post-process features
            features = self.postprocessor.normalize_features(features)
            
            logger.info(f"Extracted features shape: {features.shape}")
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction failed for {descriptor_name}: {e}")
            return None
    
    def evaluate_descriptor(self,
                           descriptor_name: str,
                           X_train: np.ndarray,
                           y_train: np.ndarray,
                           X_test: np.ndarray,
                           y_test: np.ndarray,
                           encoder_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate a single descriptor.
        
        Parameters:
        -----------
        descriptor_name : str
            Name of descriptor
        X_train, y_train : np.ndarray
            Training data
        X_test, y_test : np.ndarray
            Test data
        encoder_name : str, optional
            Name of encoder (for local descriptors)
            
        Returns:
        --------
        Dict
            Evaluation results
        """
        eval_name = f"{descriptor_name}+{encoder_name}" if encoder_name else descriptor_name
        logger.info(f"Evaluating {eval_name}...")
        
        start_time = time.time()
        
        # Extract features
        train_features = self.extract_features(X_train, descriptor_name, encoder_name)
        test_features = self.extract_features(X_test, descriptor_name, encoder_name)
        
        if train_features is None or test_features is None:
            return {
                'status': 'failed',
                'error': 'Feature extraction failed'
            }
        
        feature_extraction_time = time.time() - start_time
        
        # Evaluate with different classifiers
        classifier_results = {}
        
        for clf_name, classifier in self.classifiers.items():
            logger.info(f"Training {clf_name} classifier for {eval_name}...")
            
            try:
                clf_start_time = time.time()
                
                # Train classifier
                classifier.fit(train_features, y_train)
                
                # Predict
                y_pred = classifier.predict(test_features)
                y_proba = classifier.predict_proba(test_features) if hasattr(classifier, 'predict_proba') else None
                
                clf_time = time.time() - clf_start_time
                
                # Compute metrics
                test_metrics = self.metrics.compute_classification_metrics(y_test, y_pred, y_proba)
                
                # Cross-validation on training set
                cv_scores = self.cv_evaluator.cross_validate_classifier(
                    classifier, train_features, y_train, cv_folds=5
                )
                
                classifier_results[clf_name] = {
                    'training_time': clf_time,
                    'test_metrics': test_metrics,
                    'cv_scores': cv_scores,
                    'status': 'success'
                }
                
                logger.info(f"✓ {clf_name}: accuracy={test_metrics['accuracy']:.3f}, "
                           f"f1={test_metrics['macro_f1']:.3f}")
                
            except Exception as e:
                logger.error(f"✗ {clf_name} failed for {eval_name}: {e}")
                classifier_results[clf_name] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        # Feature info
        feature_info = {
            'dimensions': train_features.shape[1],
            'extraction_time': feature_extraction_time,
            'avg_time_per_image': feature_extraction_time / len(X_train),
            'sparsity': np.mean(train_features == 0),
            'value_range': [float(np.min(train_features)), float(np.max(train_features))]
        }
        
        total_time = time.time() - start_time
        
        return {
            'descriptor_name': descriptor_name,
            'encoder_name': encoder_name,
            'evaluation_time': total_time,
            'feature_info': feature_info,
            'classifiers': classifier_results,
            'status': 'success'
        }
    
    def evaluate_robustness(self,
                          descriptor_name: str,
                          X_test: np.ndarray,
                          y_test: np.ndarray,
                          encoder_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate robustness of descriptor.
        
        Parameters:
        -----------
        descriptor_name : str
            Name of descriptor
        X_test, y_test : np.ndarray
            Test data
        encoder_name : str, optional
            Name of encoder
            
        Returns:
        --------
        Dict
            Robustness results
        """
        eval_name = f"{descriptor_name}+{encoder_name}" if encoder_name else descriptor_name
        logger.info(f"Evaluating robustness for {eval_name}...")
        
        try:
            # Use subset for robustness testing to save time
            subset_size = min(500, len(X_test))
            indices = np.random.choice(len(X_test), subset_size, replace=False)
            X_subset = X_test[indices]
            y_subset = y_test[indices]
            
            # Extract baseline features
            baseline_features = self.extract_features(X_subset, descriptor_name, encoder_name)
            if baseline_features is None:
                return {'status': 'failed', 'error': 'Feature extraction failed'}
            
            # Evaluate robustness
            results = self.robustness_evaluator.evaluate_robustness(
                X_subset, y_subset, baseline_features,
                extract_features_func=lambda x: self.extract_features(x, descriptor_name, encoder_name)
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Robustness evaluation failed for {eval_name}: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def run_evaluation(self,
                      descriptor_names: Optional[List[str]] = None,
                      include_robustness: bool = True) -> Dict[str, Any]:
        """
        Run complete evaluation.
        
        Parameters:
        -----------
        descriptor_names : List[str], optional
            Specific descriptors to evaluate
        include_robustness : bool
            Whether to include robustness testing
            
        Returns:
        --------
        Dict
            Complete evaluation results
        """
        # Load test data
        X_test, y_test, X_train, y_train = self.load_test_data()
        
        # Determine which descriptors to evaluate
        if descriptor_names is None:
            # Find all available descriptors
            descriptor_names = []
            for desc_name in self.descriptor_classes.keys():
                desc_path = self.cache_dir / f"{desc_name}_descriptor.pkl"
                if desc_path.exists():
                    descriptor_names.append(desc_name)
        
        if not descriptor_names:
            logger.error("No trained descriptors found. Please run training first.")
            return {}
        
        logger.info(f"Evaluating descriptors: {descriptor_names}")
        
        results = {
            'evaluation_config': {
                'descriptors': descriptor_names,
                'n_train_samples': len(X_train),
                'n_test_samples': len(X_test),
                'n_classes': len(np.unique(y_test)),
                'include_robustness': include_robustness,
                'class_names': CLASS_NAMES
            },
            'descriptors': {},
            'robustness': {}
        }
        
        # Evaluate each descriptor
        for desc_name in descriptor_names:
            logger.info(f"\n{'='*50}")
            logger.info(f"EVALUATING: {desc_name.upper()}")
            logger.info(f"{'='*50}")
            
            # Check if this is a local descriptor
            is_local = desc_name in ['sift', 'orb', 'brisk', 'surf']
            
            if is_local:
                # Evaluate with each encoder
                encoder_names = ['bow', 'vlad', 'fisher']
                desc_results = {}
                
                for enc_name in encoder_names:
                    # Check if encoder exists
                    enc_path = self.cache_dir / f"{desc_name}_{enc_name}_encoder.pkl"
                    if not enc_path.exists():
                        logger.warning(f"Encoder not found: {enc_path}")
                        continue
                    
                    # Evaluate descriptor+encoder combination
                    eval_result = self.evaluate_descriptor(
                        desc_name, X_train, y_train, X_test, y_test, enc_name
                    )
                    desc_results[enc_name] = eval_result
                    
                    # Robustness evaluation
                    if include_robustness and eval_result.get('status') == 'success':
                        robustness_result = self.evaluate_robustness(
                            desc_name, X_test, y_test, enc_name
                        )
                        results['robustness'][f"{desc_name}+{enc_name}"] = robustness_result
                
                results['descriptors'][desc_name] = desc_results
                
            else:
                # Global descriptor - evaluate directly
                eval_result = self.evaluate_descriptor(
                    desc_name, X_train, y_train, X_test, y_test
                )
                results['descriptors'][desc_name] = eval_result
                
                # Robustness evaluation
                if include_robustness and eval_result.get('status') == 'success':
                    robustness_result = self.evaluate_robustness(
                        desc_name, X_test, y_test
                    )
                    results['robustness'][desc_name] = robustness_result
        
        return results
    
    def save_results(self, results: Dict[str, Any], filename: str = "evaluation_results.json") -> None:
        """
        Save evaluation results.
        
        Parameters:
        -----------
        results : Dict
            Evaluation results
        filename : str
            Output filename
        """
        results_path = self.results_dir / filename
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to: {results_path}")
    
    def generate_report(self, results: Dict[str, Any]) -> None:
        """
        Generate comprehensive evaluation report.
        
        Parameters:
        -----------
        results : Dict
            Evaluation results
        """
        report_path = self.results_dir / "evaluation_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("UNSUPERVISED DESCRIPTORS EVALUATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            # Configuration
            config = results.get('evaluation_config', {})
            f.write("Configuration:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Descriptors evaluated: {len(config.get('descriptors', []))}\n")
            f.write(f"Training samples: {config.get('n_train_samples', 'N/A')}\n")
            f.write(f"Test samples: {config.get('n_test_samples', 'N/A')}\n")
            f.write(f"Number of classes: {config.get('n_classes', 'N/A')}\n\n")
            
            # Performance summary
            f.write("PERFORMANCE SUMMARY\n")
            f.write("=" * 30 + "\n\n")
            
            descriptor_results = results.get('descriptors', {})
            
            # Collect best results
            best_results = []
            
            for desc_name, desc_result in descriptor_results.items():
                if isinstance(desc_result, dict):
                    # Check if this has encoders (local descriptor)
                    if any(key in ['bow', 'vlad', 'fisher'] for key in desc_result.keys()):
                        # Local descriptor with encoders
                        for enc_name, enc_result in desc_result.items():
                            if enc_result.get('status') == 'success':
                                classifiers = enc_result.get('classifiers', {})
                                best_acc = 0
                                best_f1 = 0
                                for clf_result in classifiers.values():
                                    if clf_result.get('status') == 'success':
                                        metrics = clf_result.get('test_metrics', {})
                                        acc = metrics.get('accuracy', 0)
                                        f1 = metrics.get('macro_f1', 0)
                                        best_acc = max(best_acc, acc)
                                        best_f1 = max(best_f1, f1)
                                
                                if best_acc > 0:
                                    best_results.append({
                                        'name': f"{desc_name}+{enc_name}",
                                        'accuracy': best_acc,
                                        'f1': best_f1,
                                        'dimensions': enc_result.get('feature_info', {}).get('dimensions', 0)
                                    })
                    else:
                        # Global descriptor
                        if desc_result.get('status') == 'success':
                            classifiers = desc_result.get('classifiers', {})
                            best_acc = 0
                            best_f1 = 0
                            for clf_result in classifiers.values():
                                if clf_result.get('status') == 'success':
                                    metrics = clf_result.get('test_metrics', {})
                                    acc = metrics.get('accuracy', 0)
                                    f1 = metrics.get('macro_f1', 0)
                                    best_acc = max(best_acc, acc)
                                    best_f1 = max(best_f1, f1)
                            
                            if best_acc > 0:
                                best_results.append({
                                    'name': desc_name,
                                    'accuracy': best_acc,
                                    'f1': best_f1,
                                    'dimensions': desc_result.get('feature_info', {}).get('dimensions', 0)
                                })
            
            # Sort by accuracy
            best_results.sort(key=lambda x: x['accuracy'], reverse=True)
            
            f.write("Top Performing Descriptors:\n")
            f.write("-" * 30 + "\n")
            f.write(f"{'Descriptor':<25} {'Accuracy':<10} {'F1-Score':<10} {'Dims':<8}\n")
            f.write("-" * 55 + "\n")
            
            for result in best_results[:10]:  # Top 10
                f.write(f"{result['name']:<25} {result['accuracy']:<10.3f} "
                       f"{result['f1']:<10.3f} {result['dimensions']:<8}\n")
            
            f.write("\n")
            
            # Robustness summary
            robustness_results = results.get('robustness', {})
            if robustness_results:
                f.write("ROBUSTNESS ANALYSIS\n")
                f.write("=" * 25 + "\n\n")
                
                for desc_name, rob_result in robustness_results.items():
                    if rob_result.get('status') == 'success':
                        f.write(f"{desc_name}:\n")
                        for transform, result in rob_result.items():
                            if transform != 'baseline' and isinstance(result, dict):
                                drop = result.get('drop', 0)
                                f.write(f"  {transform}: {drop:.3f} accuracy drop\n")
                        f.write("\n")
        
        logger.info(f"Report saved to: {report_path}")
    
    def create_visualizations(self, results: Dict[str, Any]) -> None:
        """
        Create visualization plots.
        
        Parameters:
        -----------
        results : Dict
            Evaluation results
        """
        logger.info("Creating visualizations...")
        
        # Performance comparison plot
        viz_dir = self.results_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        try:
            # Prepare data for visualization
            viz_results = {}
            descriptor_results = results.get('descriptors', {})
            
            for desc_name, desc_result in descriptor_results.items():
                if isinstance(desc_result, dict):
                    if any(key in ['bow', 'vlad', 'fisher'] for key in desc_result.keys()):
                        # Local descriptor - take best encoder
                        best_acc = 0
                        best_result = None
                        for enc_name, enc_result in desc_result.items():
                            if enc_result.get('status') == 'success':
                                classifiers = enc_result.get('classifiers', {})
                                for clf_result in classifiers.values():
                                    if clf_result.get('status') == 'success':
                                        acc = clf_result.get('test_metrics', {}).get('accuracy', 0)
                                        if acc > best_acc:
                                            best_acc = acc
                                            best_result = enc_result
                        
                        if best_result:
                            viz_results[desc_name] = best_result
                    else:
                        # Global descriptor
                        if desc_result.get('status') == 'success':
                            viz_results[desc_name] = desc_result
            
            # Performance comparison
            if viz_results:
                self.visualizer.plot_performance_comparison(
                    viz_results, 
                    save_path=str(viz_dir / "performance_comparison.png")
                )
            
            # Efficiency analysis
            if viz_results:
                self.visualizer.plot_efficiency_analysis(
                    viz_results,
                    save_path=str(viz_dir / "efficiency_analysis.png")
                )
            
            # Robustness analysis
            robustness_results = results.get('robustness', {})
            if robustness_results:
                self.visualizer.plot_robustness_analysis(
                    robustness_results,
                    save_path=str(viz_dir / "robustness_analysis.png")
                )
            
            logger.info(f"Visualizations saved to: {viz_dir}")
            
        except Exception as e:
            logger.error(f"Failed to create visualizations: {e}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate unsupervised descriptors on STL-10')
    parser.add_argument('--data-dir', type=str, default='data/',
                       help='Directory containing STL-10 dataset')
    parser.add_argument('--cache-dir', type=str, default='cache/',
                       help='Directory with trained models')
    parser.add_argument('--results-dir', type=str, default='results/',
                       help='Directory to save results')
    parser.add_argument('--descriptors', type=str, nargs='+',
                       help='Specific descriptors to evaluate')
    parser.add_argument('--no-robustness', action='store_true',
                       help='Skip robustness evaluation')
    parser.add_argument('--no-visualizations', action='store_true',
                       help='Skip creating visualizations')
    
    args = parser.parse_args()
    
    # Validate directories
    if not os.path.exists(args.data_dir):
        logger.error(f"Data directory not found: {args.data_dir}")
        return 1
    
    if not os.path.exists(args.cache_dir):
        logger.error(f"Cache directory not found: {args.cache_dir}")
        logger.error("Please run training first: python scripts/train_descriptors.py")
        return 1
    
    try:
        # Initialize evaluator
        evaluator = DescriptorEvaluator(
            data_dir=args.data_dir,
            cache_dir=args.cache_dir,
            results_dir=args.results_dir
        )
        
        # Run evaluation
        logger.info("Starting evaluation...")
        results = evaluator.run_evaluation(
            descriptor_names=args.descriptors,
            include_robustness=not args.no_robustness
        )
        
        if not results:
            logger.error("Evaluation failed or no results generated")
            return 1
        
        # Save results
        evaluator.save_results(results)
        
        # Generate report
        evaluator.generate_report(results)
        
        # Create visualizations
        if not args.no_visualizations:
            evaluator.create_visualizations(results)
        
        print("\n" + "="*60)
        print("EVALUATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Results saved to: {args.results_dir}")
        print("Files generated:")
        print("  - evaluation_results.json (detailed results)")
        print("  - evaluation_report.txt (summary report)")
        if not args.no_visualizations:
            print("  - visualizations/ (plots and charts)")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())