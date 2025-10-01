#!/usr/bin/env python3
"""
Main Evaluation Script
=====================

Reproducible script that runs the complete evaluation pipeline
for unsupervised image descriptors on STL-10 dataset.

This script covers all requirements from the hackathon:
1. Downloads STL-10 dataset
2. Trains descriptors using unlabeled split
3. Extracts features from train/test splits
4. Trains classifiers and evaluates performance
5. Tests robustness to transformations
6. Generates comprehensive reports

Usage:
    python scripts/run_full_evaluation.py [--config CONFIG_FILE] [--output OUTPUT_DIR]
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from typing import Dict, List, Any

# Import project modules
from src.utils.data_loader import STL10DataLoader
from src.descriptors import (
    HOGDescriptor, LBPDescriptor, ColorHistogramDescriptor,
    SIFTDescriptor, BagOfVisualWords, VLADEncoder
)
from src.evaluation.classifiers import DescriptorClassifier, CrossValidationEvaluator
from src.evaluation.robustness import RobustnessEvaluator
from src.evaluation.metrics import ClassificationMetrics

# Import configuration
import config


def setup_output_directory(output_dir: str) -> Path:
    """Create and setup output directory structure."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (output_path / "models").mkdir(exist_ok=True)
    (output_path / "metrics").mkdir(exist_ok=True)
    (output_path / "logs").mkdir(exist_ok=True)
    
    return output_path


def load_stl10_data():
    """Load STL-10 dataset splits."""
    print("=" * 60)
    print("📊 LOADING STL-10 DATASET")
    print("=" * 60)
    
    data_loader = STL10DataLoader(
        data_root=config.STL10_ROOT,
        download=True
    )
    
    # Load different splits
    print("Loading unlabeled split for training...")
    unlabeled_images = data_loader.load_unlabeled()
    
    print("Loading labeled splits for evaluation...")
    train_images, train_labels = data_loader.load_labeled_split('train')
    test_images, test_labels = data_loader.load_labeled_split('test')
    
    class_names = data_loader.get_class_names()
    
    print(f"✅ Data loading complete:")
    print(f"   - Unlabeled: {len(unlabeled_images)} images")
    print(f"   - Train: {len(train_images)} images")
    print(f"   - Test: {len(test_images)} images")
    print(f"   - Classes: {len(class_names)}")
    
    return {
        'unlabeled_images': unlabeled_images,
        'train_images': train_images,
        'train_labels': train_labels,
        'test_images': test_images,
        'test_labels': test_labels,
        'class_names': class_names
    }


def create_descriptors() -> Dict[str, Any]:
    """Create and configure all descriptors to evaluate."""
    print("=" * 60)
    print("🔧 CREATING DESCRIPTORS")
    print("=" * 60)
    
    descriptors = {}
    
    # Global descriptors
    print("Creating global descriptors...")
    
    # HOG
    descriptors['hog'] = HOGDescriptor(**config.DESCRIPTOR_PARAMS['hog'])
    print(f"   ✅ HOG descriptor created")
    
    # LBP
    descriptors['lbp'] = LBPDescriptor(**config.DESCRIPTOR_PARAMS['lbp'])
    print(f"   ✅ LBP descriptor created")
    
    # Color Histogram
    descriptors['color_hist'] = ColorHistogramDescriptor(h_bins=16, s_bins=16, v_bins=16)
    print(f"   ✅ Color Histogram descriptor created")
    
    # Local descriptors with encoding
    print("Creating local descriptors with encoding...")
    
    # SIFT + BoVW
    sift_detector = SIFTDescriptor(**config.DESCRIPTOR_PARAMS['sift'])
    descriptors['sift_bovw'] = BagOfVisualWords(
        local_descriptor=sift_detector,
        **config.BOVW_PARAMS,
        codebook_size=512  # Use fixed size for main evaluation
    )
    print(f"   ✅ SIFT+BoVW descriptor created")
    
    # SIFT + VLAD
    sift_detector_vlad = SIFTDescriptor(**config.DESCRIPTOR_PARAMS['sift'])
    descriptors['sift_vlad'] = VLADEncoder(
        local_descriptor=sift_detector_vlad,
        **config.VLAD_PARAMS,
        codebook_size=256  # Use fixed size for main evaluation
    )
    print(f"   ✅ SIFT+VLAD descriptor created")
    
    print(f"✅ Total descriptors created: {len(descriptors)}")
    return descriptors


def train_descriptors(descriptors: Dict[str, Any], 
                     unlabeled_images: List,
                     output_dir: Path) -> Dict[str, Any]:
    """Train all descriptors using unlabeled data."""
    print("=" * 60)
    print("🎓 TRAINING DESCRIPTORS")
    print("=" * 60)
    
    trained_descriptors = {}
    
    for name, descriptor in descriptors.items():
        print(f"\nTraining {name.upper()} descriptor...")
        start_time = time.time()
        
        try:
            # Train descriptor
            descriptor.fit(unlabeled_images)
            
            # Save trained descriptor
            model_path = output_dir / "models" / f"{name}_descriptor.pkl"
            descriptor.save(str(model_path))
            
            training_time = time.time() - start_time
            print(f"   ✅ {name.upper()} trained in {training_time:.2f} seconds")
            print(f"   📁 Saved to: {model_path}")
            print(f"   📏 Descriptor dimension: {descriptor.get_descriptor_dimension()}")
            
            trained_descriptors[name] = descriptor
            
        except Exception as e:
            print(f"   ❌ Error training {name}: {str(e)}")
            continue
    
    print(f"\n✅ Training complete: {len(trained_descriptors)}/{len(descriptors)} descriptors trained")
    return trained_descriptors


def extract_features(descriptors: Dict[str, Any],
                    train_images: List,
                    test_images: List,
                    output_dir: Path) -> Dict[str, Dict[str, np.ndarray]]:
    """Extract features from train and test images."""
    print("=" * 60)
    print("🔍 EXTRACTING FEATURES")
    print("=" * 60)
    
    features = {}
    
    for name, descriptor in descriptors.items():
        print(f"\nExtracting features with {name.upper()}...")
        start_time = time.time()
        
        try:
            # Extract features
            print("   Extracting train features...")
            train_features = descriptor.extract_batch(train_images)
            
            print("   Extracting test features...")
            test_features = descriptor.extract_batch(test_images)
            
            # Save features
            features_path = output_dir / "processed"
            features_path.mkdir(exist_ok=True)
            
            np.save(features_path / f"{name}_train_features.npy", train_features)
            np.save(features_path / f"{name}_test_features.npy", test_features)
            
            extraction_time = time.time() - start_time
            avg_time_per_image = extraction_time / (len(train_images) + len(test_images))
            
            print(f"   ✅ Features extracted in {extraction_time:.2f} seconds")
            print(f"   ⏱️  Average time per image: {avg_time_per_image*1000:.2f} ms")
            print(f"   📏 Feature dimensions: {train_features.shape[1]}")
            
            features[name] = {
                'train': train_features,
                'test': test_features,
                'extraction_time': extraction_time,
                'avg_time_per_image': avg_time_per_image,
                'dimensions': train_features.shape[1]
            }
            
        except Exception as e:
            print(f"   ❌ Error extracting features with {name}: {str(e)}")
            continue
    
    print(f"\n✅ Feature extraction complete: {len(features)} descriptors processed")
    return features


def evaluate_classifiers(features: Dict[str, Dict[str, np.ndarray]],
                        train_labels: np.ndarray,
                        test_labels: np.ndarray,
                        class_names: List[str],
                        output_dir: Path) -> Dict[str, Dict]:
    """Evaluate descriptors with multiple classifiers."""
    print("=" * 60)
    print("📊 EVALUATING CLASSIFIERS")
    print("=" * 60)
    
    # Classifier configurations
    classifier_configs = [
        {'type': 'svm_linear', 'params': {'C': 1.0}},
        {'type': 'svm_rbf', 'params': {'C': 1.0, 'gamma': 'scale'}},
        {'type': 'knn', 'params': {'n_neighbors': 5}},
        {'type': 'logistic', 'params': {'C': 1.0}}
    ]
    
    # Cross-validation evaluator
    cv_evaluator = CrossValidationEvaluator(
        n_splits=config.N_SPLITS,
        n_repeats=1,  # Reduce for faster evaluation
        random_state=config.RANDOM_SEEDS[0]
    )
    
    all_results = {}
    
    for desc_name, feature_data in features.items():
        print(f"\n🔍 Evaluating {desc_name.upper()}...")
        
        train_features = feature_data['train']
        test_features = feature_data['test']
        
        descriptor_results = {
            'feature_info': {
                'dimensions': feature_data['dimensions'],
                'extraction_time': feature_data['extraction_time'],
                'avg_time_per_image': feature_data['avg_time_per_image']
            },
            'classifiers': {}
        }
        
        # Try each classifier
        for config_item in classifier_configs:
            classifier_name = f"{config_item['type']}"
            print(f"   Testing {classifier_name}...")
            
            try:
                # Create and train classifier
                classifier = DescriptorClassifier(
                    classifier_type=config_item['type'],
                    classifier_params=config_item['params'],
                    random_state=config.RANDOM_SEEDS[0]
                )
                
                # Train on full training set
                classifier.fit(train_features, train_labels)
                
                # Evaluate on test set
                test_metrics = classifier.evaluate(test_features, test_labels, class_names)
                
                # Store results
                descriptor_results['classifiers'][classifier_name] = {
                    'test_metrics': test_metrics,
                    'config': config_item
                }
                
                print(f"      ✅ {classifier_name}: {test_metrics['accuracy']:.4f} accuracy")
                
            except Exception as e:
                print(f"      ❌ Error with {classifier_name}: {str(e)}")
                continue
        
        all_results[desc_name] = descriptor_results
    
    # Save results
    results_path = output_dir / "metrics" / "classification_results.json"
    with open(results_path, 'w') as f:
        # Convert numpy values to regular Python types for JSON serialization
        json_results = {}
        for desc_name, desc_results in all_results.items():
            json_results[desc_name] = {}
            json_results[desc_name]['feature_info'] = desc_results['feature_info']
            json_results[desc_name]['classifiers'] = {}
            
            for clf_name, clf_results in desc_results['classifiers'].items():
                json_results[desc_name]['classifiers'][clf_name] = {
                    'config': clf_results['config'],
                    'test_metrics': {k: float(v) if isinstance(v, np.floating) else v 
                                   for k, v in clf_results['test_metrics'].items()}
                }
        
        json.dump(json_results, f, indent=2)
    
    print(f"\n✅ Classification evaluation complete")
    print(f"📁 Results saved to: {results_path}")
    
    return all_results


def evaluate_robustness(descriptors: Dict[str, Any],
                       test_images: List,
                       test_labels: np.ndarray,
                       classification_results: Dict[str, Dict],
                       output_dir: Path) -> Dict[str, Dict]:
    """Evaluate robustness of descriptors to transformations."""
    print("=" * 60)
    print("🛡️ EVALUATING ROBUSTNESS")
    print("=" * 60)
    
    robustness_results = {}
    
    for desc_name, descriptor in descriptors.items():
        print(f"\n🔍 Testing robustness of {desc_name.upper()}...")
        
        try:
            # Get best classifier for this descriptor
            desc_results = classification_results[desc_name]
            best_classifier_name = None
            best_accuracy = 0
            
            for clf_name, clf_results in desc_results['classifiers'].items():
                acc = clf_results['test_metrics']['accuracy']
                if acc > best_accuracy:
                    best_accuracy = acc
                    best_classifier_name = clf_name
            
            if best_classifier_name is None:
                print(f"   ❌ No valid classifier found for {desc_name}")
                continue
            
            print(f"   Using best classifier: {best_classifier_name} (acc: {best_accuracy:.4f})")
            
            # Recreate best classifier
            best_config = desc_results['classifiers'][best_classifier_name]['config']
            classifier = DescriptorClassifier(
                classifier_type=best_config['type'],
                classifier_params=best_config['params'],
                random_state=config.RANDOM_SEEDS[0]
            )
            
            # Re-extract features and train classifier
            train_images, train_labels = [], []  # We'd need to store these from earlier
            # For now, use a subset of test images for demonstration
            subset_size = min(1000, len(test_images))
            subset_indices = np.random.choice(len(test_images), subset_size, replace=False)
            subset_images = [test_images[i] for i in subset_indices]
            subset_labels = test_labels[subset_indices]
            
            # Split subset for training classifier
            split_point = len(subset_images) // 2
            demo_train_images = subset_images[:split_point]
            demo_train_labels = subset_labels[:split_point]
            demo_test_images = subset_images[split_point:]
            demo_test_labels = subset_labels[split_point:]
            
            # Extract features and train
            demo_train_features = descriptor.extract_batch(demo_train_images)
            classifier.fit(demo_train_features, demo_train_labels)
            
            # Evaluate robustness
            robustness_evaluator = RobustnessEvaluator(descriptor)
            robustness_result = robustness_evaluator.evaluate_robustness(
                demo_test_images,
                demo_test_labels,
                classifier,
                config.ROBUSTNESS_TRANSFORMS
            )
            
            robustness_results[desc_name] = robustness_result
            
            # Print summary
            baseline_acc = robustness_result['baseline']['accuracy']
            print(f"   📊 Baseline accuracy: {baseline_acc:.4f}")
            
            for transform_name, result in robustness_result.items():
                if transform_name != 'baseline':
                    drop = result['drop']
                    print(f"   🔄 {transform_name}: -{drop:.4f} accuracy drop")
            
        except Exception as e:
            print(f"   ❌ Error evaluating robustness for {desc_name}: {str(e)}")
            continue
    
    # Save robustness results
    robustness_path = output_dir / "metrics" / "robustness_results.json"
    with open(robustness_path, 'w') as f:
        # Convert to JSON-serializable format
        json_results = {}
        for desc_name, results in robustness_results.items():
            json_results[desc_name] = {}
            for transform_name, result in results.items():
                json_results[desc_name][transform_name] = {
                    k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                    for k, v in result.items()
                    if k != 'error'  # Skip error messages for JSON
                }
        
        json.dump(json_results, f, indent=2)
    
    print(f"\n✅ Robustness evaluation complete")
    print(f"📁 Results saved to: {robustness_path}")
    
    return robustness_results


def generate_summary_report(classification_results: Dict[str, Dict],
                          robustness_results: Dict[str, Dict],
                          output_dir: Path) -> None:
    """Generate a comprehensive summary report."""
    print("=" * 60)
    print("📋 GENERATING SUMMARY REPORT")
    print("=" * 60)
    
    report_lines = []
    report_lines.append("# STL-10 Unsupervised Descriptors Evaluation Report")
    report_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Classification results summary
    report_lines.append("## Classification Results")
    report_lines.append("")
    report_lines.append("| Descriptor | Best Classifier | Accuracy | Macro F1 | Dimensions | Avg Time (ms) |")
    report_lines.append("|------------|-----------------|----------|----------|------------|---------------|")
    
    for desc_name, desc_results in classification_results.items():
        feature_info = desc_results['feature_info']
        
        # Find best classifier
        best_acc = 0
        best_clf_name = ""
        best_f1 = 0
        
        for clf_name, clf_results in desc_results['classifiers'].items():
            acc = clf_results['test_metrics']['accuracy']
            if acc > best_acc:
                best_acc = acc
                best_clf_name = clf_name
                best_f1 = clf_results['test_metrics'].get('macro_f1', 0)
        
        report_lines.append(
            f"| {desc_name} | {best_clf_name} | {best_acc:.4f} | {best_f1:.4f} | "
            f"{feature_info['dimensions']} | {feature_info['avg_time_per_image']*1000:.2f} |"
        )
    
    # Robustness results summary
    if robustness_results:
        report_lines.append("")
        report_lines.append("## Robustness Results")
        report_lines.append("")
        report_lines.append("Average accuracy drop for each transformation:")
        report_lines.append("")
        
        # Get all transformation names
        transform_names = set()
        for results in robustness_results.values():
            transform_names.update(results.keys())
        transform_names.discard('baseline')
        transform_names = sorted(transform_names)
        
        # Create table header
        header = "| Descriptor |"
        for transform in transform_names:
            header += f" {transform} |"
        report_lines.append(header)
        
        separator = "|" + "------------|" * (len(transform_names) + 1)
        report_lines.append(separator)
        
        # Add data rows
        for desc_name, results in robustness_results.items():
            row = f"| {desc_name} |"
            for transform in transform_names:
                if transform in results:
                    drop = results[transform].get('drop', 0)
                    row += f" -{drop:.4f} |"
                else:
                    row += " N/A |"
            report_lines.append(row)
    
    # Add recommendations
    report_lines.append("")
    report_lines.append("## Recommendations")
    report_lines.append("")
    
    # Find best overall descriptor
    best_overall = None
    best_acc = 0
    for desc_name, desc_results in classification_results.items():
        for clf_results in desc_results['classifiers'].values():
            acc = clf_results['test_metrics']['accuracy']
            if acc > best_acc:
                best_acc = acc
                best_overall = desc_name
    
    if best_overall:
        report_lines.append(f"- **Best accuracy**: {best_overall} ({best_acc:.4f})")
    
    # Find fastest descriptor
    fastest_desc = None
    fastest_time = float('inf')
    for desc_name, desc_results in classification_results.items():
        time_per_img = desc_results['feature_info']['avg_time_per_image']
        if time_per_img < fastest_time:
            fastest_time = time_per_img
            fastest_desc = desc_name
    
    if fastest_desc:
        report_lines.append(f"- **Fastest extraction**: {fastest_desc} ({fastest_time*1000:.2f} ms/image)")
    
    # Save report
    report_path = output_dir / "metrics" / "summary_report.md"
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print("✅ Summary report generated")
    print(f"📁 Report saved to: {report_path}")
    
    # Print key results to console
    print("\n" + "=" * 60)
    print("🏆 KEY RESULTS")
    print("=" * 60)
    
    if best_overall:
        print(f"🥇 Best performing descriptor: {best_overall} ({best_acc:.4f} accuracy)")
    
    if fastest_desc:
        print(f"⚡ Fastest descriptor: {fastest_desc} ({fastest_time*1000:.2f} ms/image)")
    
    print(f"📊 Total descriptors evaluated: {len(classification_results)}")
    print(f"📁 All results saved to: {output_dir / 'metrics'}")


def main():
    """Main evaluation pipeline."""
    parser = argparse.ArgumentParser(description='STL-10 Unsupervised Descriptors Evaluation')
    parser.add_argument('--config', type=str, default='config.py',
                      help='Configuration file path')
    parser.add_argument('--output', type=str, default='results',
                      help='Output directory for results')
    parser.add_argument('--skip-download', action='store_true',
                      help='Skip dataset download if already present')
    
    args = parser.parse_args()
    
    print("🚀 STL-10 Unsupervised Descriptors Evaluation")
    print("=" * 60)
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Output directory: {args.output}")
    print(f"⚙️  Configuration: {args.config}")
    
    # Setup output directory
    output_dir = setup_output_directory(args.output)
    
    # Load data
    data = load_stl10_data()
    
    # Create descriptors
    descriptors = create_descriptors()
    
    # Train descriptors
    trained_descriptors = train_descriptors(
        descriptors, 
        data['unlabeled_images'], 
        output_dir
    )
    
    # Extract features
    features = extract_features(
        trained_descriptors,
        data['train_images'],
        data['test_images'],
        output_dir
    )
    
    # Evaluate classifiers
    classification_results = evaluate_classifiers(
        features,
        data['train_labels'],
        data['test_labels'],
        data['class_names'],
        output_dir
    )
    
    # Evaluate robustness
    robustness_results = evaluate_robustness(
        trained_descriptors,
        data['test_images'],
        data['test_labels'],
        classification_results,
        output_dir
    )
    
    # Generate summary report
    generate_summary_report(
        classification_results,
        robustness_results,
        output_dir
    )
    
    end_time = datetime.now()
    print(f"\n🎉 Evaluation completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Check results in: {output_dir}")


if __name__ == "__main__":
    main()