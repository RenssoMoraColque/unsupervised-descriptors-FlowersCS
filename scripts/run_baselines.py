#!/usr/bin/env python3
"""
Baseline Descriptors Evaluation
===============================

Quick evaluation script that implements and tests the three baseline
descriptors suggested in the hackathon description:

1. Baseline A (rápida): HOG (3780 dim) → PCA(512) → SVM lineal
2. Baseline B (BoVW): SIFT (dense) → k-means K=512 → BoVW histograma L2 → SVM
3. Baseline C (global simple): Color histogram (HSV 64 bins) + LBP → concatenado → k-NN

This script provides a quick way to get baseline results before running
the full evaluation pipeline.

Usage:
    python scripts/run_baselines.py
"""

import os
import sys
import time
from datetime import datetime

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

from src.utils.data_loader import STL10DataLoader
from src.descriptors import HOGDescriptor, LBPDescriptor, ColorHistogramDescriptor, SIFTDescriptor, BagOfVisualWords
import config


def create_baseline_a():
    """
    Baseline A: HOG (3780 dim) → PCA(512) → SVM lineal
    """
    print("Creating Baseline A: HOG + PCA + SVM Linear")
    
    # HOG descriptor
    hog_descriptor = HOGDescriptor(
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        orientations=9
    )
    
    # PCA + SVM pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=512, random_state=42)),
        ('svm', SVC(kernel='linear', C=1.0, random_state=42))
    ])
    
    return hog_descriptor, pipeline


def create_baseline_b():
    """
    Baseline B: SIFT (dense) → k-means K=512 → BoVW histograma L2 → SVM
    """
    print("Creating Baseline B: SIFT + BoVW + SVM")
    
    # SIFT detector
    sift_detector = SIFTDescriptor(**config.DESCRIPTOR_PARAMS['sift'])
    
    # BoVW encoder
    bovw_descriptor = BagOfVisualWords(
        local_descriptor=sift_detector,
        codebook_size=512,
        max_descriptors_per_image=100,
        random_state=42
    )
    
    # SVM classifier
    classifier = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42))
    ])
    
    return bovw_descriptor, classifier


def create_baseline_c():
    """
    Baseline C: Color histogram (HSV 64 bins) + LBP → concatenado → k-NN
    """
    print("Creating Baseline C: Color Histogram + LBP + k-NN")
    
    # Color histogram (HSV with total 64 bins)
    color_descriptor = ColorHistogramDescriptor(h_bins=16, s_bins=16, v_bins=16)
    
    # LBP descriptor
    lbp_descriptor = LBPDescriptor(
        radius=3,
        n_points=24,
        method='uniform'
    )
    
    # k-NN classifier
    classifier = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=5, weights='distance'))
    ])
    
    return color_descriptor, lbp_descriptor, classifier


def concatenate_descriptors(desc1_features, desc2_features):
    """Concatenate features from two descriptors."""
    return np.concatenate([desc1_features, desc2_features], axis=1)


def evaluate_baseline(name, descriptor, classifier, train_images, train_labels, 
                     test_images, test_labels, class_names, unlabeled_images=None):
    """Evaluate a single baseline configuration."""
    print(f"\n{'='*50}")
    print(f"🔍 EVALUATING {name}")
    print(f"{'='*50}")
    
    start_time = time.time()
    
    try:
        # Train descriptor if needed
        if hasattr(descriptor, 'fit') and unlabeled_images is not None:
            print("Training descriptor on unlabeled data...")
            descriptor.fit(unlabeled_images)
        elif hasattr(descriptor, 'fit'):
            print("Training descriptor on training data...")
            descriptor.fit(train_images)
        
        # Extract features
        print("Extracting training features...")
        train_features = descriptor.extract_batch(train_images)
        
        print("Extracting test features...")
        test_features = descriptor.extract_batch(test_images)
        
        print(f"Feature dimensions: {train_features.shape[1]}")
        
        # Train classifier
        print("Training classifier...")
        classifier.fit(train_features, train_labels)
        
        # Evaluate
        print("Evaluating on test set...")
        test_predictions = classifier.predict(test_features)
        test_accuracy = accuracy_score(test_labels, test_predictions)
        
        # Also evaluate on training set
        train_predictions = classifier.predict(train_features)
        train_accuracy = accuracy_score(train_labels, train_predictions)
        
        total_time = time.time() - start_time
        avg_time_per_image = total_time / (len(train_images) + len(test_images))
        
        # Print results
        print(f"\n✅ {name} Results:")
        print(f"   📊 Train Accuracy: {train_accuracy:.4f}")
        print(f"   📊 Test Accuracy:  {test_accuracy:.4f}")
        print(f"   📏 Feature Dim:    {train_features.shape[1]}")
        print(f"   ⏱️  Total Time:     {total_time:.2f} seconds")
        print(f"   ⏱️  Time/Image:     {avg_time_per_image*1000:.2f} ms")
        
        # Detailed classification report
        print(f"\n📋 Detailed Classification Report for {name}:")
        print(classification_report(test_labels, test_predictions, 
                                  target_names=class_names, zero_division=0))
        
        return {
            'name': name,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'feature_dimensions': train_features.shape[1],
            'total_time': total_time,
            'avg_time_per_image': avg_time_per_image,
            'predictions': test_predictions
        }
        
    except Exception as e:
        print(f"❌ Error evaluating {name}: {str(e)}")
        return None


def evaluate_baseline_c_combined(color_desc, lbp_desc, classifier, 
                                train_images, train_labels, test_images, test_labels, 
                                class_names):
    """Special evaluation for Baseline C which combines two descriptors."""
    print(f"\n{'='*50}")
    print(f"🔍 EVALUATING BASELINE C (COMBINED)")
    print(f"{'='*50}")
    
    start_time = time.time()
    
    try:
        # Train descriptors
        print("Training color histogram descriptor...")
        color_desc.fit(train_images)
        
        print("Training LBP descriptor...")
        lbp_desc.fit(train_images)
        
        # Extract features
        print("Extracting color histogram features...")
        train_color_features = color_desc.extract_batch(train_images)
        test_color_features = color_desc.extract_batch(test_images)
        
        print("Extracting LBP features...")
        train_lbp_features = lbp_desc.extract_batch(train_images)
        test_lbp_features = lbp_desc.extract_batch(test_images)
        
        # Concatenate features
        print("Concatenating features...")
        train_features = concatenate_descriptors(train_color_features, train_lbp_features)
        test_features = concatenate_descriptors(test_color_features, test_lbp_features)
        
        print(f"Combined feature dimensions: {train_features.shape[1]}")
        print(f"  - Color histogram: {train_color_features.shape[1]} dims")
        print(f"  - LBP: {train_lbp_features.shape[1]} dims")
        
        # Train classifier
        print("Training k-NN classifier...")
        classifier.fit(train_features, train_labels)
        
        # Evaluate
        print("Evaluating on test set...")
        test_predictions = classifier.predict(test_features)
        test_accuracy = accuracy_score(test_labels, test_predictions)
        
        train_predictions = classifier.predict(train_features)
        train_accuracy = accuracy_score(train_labels, train_predictions)
        
        total_time = time.time() - start_time
        avg_time_per_image = total_time / (len(train_images) + len(test_images))
        
        # Print results
        print(f"\n✅ BASELINE C Results:")
        print(f"   📊 Train Accuracy: {train_accuracy:.4f}")
        print(f"   📊 Test Accuracy:  {test_accuracy:.4f}")
        print(f"   📏 Feature Dim:    {train_features.shape[1]}")
        print(f"   ⏱️  Total Time:     {total_time:.2f} seconds")
        print(f"   ⏱️  Time/Image:     {avg_time_per_image*1000:.2f} ms")
        
        # Detailed classification report
        print(f"\n📋 Detailed Classification Report for BASELINE C:")
        print(classification_report(test_labels, test_predictions, 
                                  target_names=class_names, zero_division=0))
        
        return {
            'name': 'Baseline C (Color + LBP + k-NN)',
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'feature_dimensions': train_features.shape[1],
            'total_time': total_time,
            'avg_time_per_image': avg_time_per_image,
            'predictions': test_predictions
        }
        
    except Exception as e:
        print(f"❌ Error evaluating Baseline C: {str(e)}")
        return None


def main():
    """Main function to run all baseline evaluations."""
    print("🚀 STL-10 Baseline Descriptors Evaluation")
    print("=" * 60)
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    print("\n📊 Loading STL-10 dataset...")
    data_loader = STL10DataLoader(data_root=config.STL10_ROOT, download=True)
    
    # Load labeled splits for evaluation
    train_images, train_labels = data_loader.load_labeled_split('train')
    test_images, test_labels = data_loader.load_labeled_split('test')
    class_names = data_loader.get_class_names()
    
    # For faster baseline evaluation, use a subset of unlabeled data
    print("Loading subset of unlabeled data for training...")
    unlabeled_images = data_loader.load_unlabeled()
    # Use only first 10k for faster baseline evaluation
    unlabeled_subset = unlabeled_images[:10000]
    
    print(f"✅ Data loaded:")
    print(f"   - Train: {len(train_images)} images")
    print(f"   - Test: {len(test_images)} images")
    print(f"   - Unlabeled subset: {len(unlabeled_subset)} images")
    print(f"   - Classes: {len(class_names)}")
    
    # Results storage
    all_results = []
    
    # Evaluate Baseline A: HOG + PCA + SVM
    try:
        hog_descriptor, hog_pipeline = create_baseline_a()
        result_a = evaluate_baseline(
            "BASELINE A (HOG + PCA + SVM Linear)",
            hog_descriptor, hog_pipeline,
            train_images, train_labels,
            test_images, test_labels,
            class_names, unlabeled_subset
        )
        if result_a:
            all_results.append(result_a)
    except Exception as e:
        print(f"❌ Failed to evaluate Baseline A: {str(e)}")
    
    # Evaluate Baseline B: SIFT + BoVW + SVM
    try:
        bovw_descriptor, bovw_pipeline = create_baseline_b()
        result_b = evaluate_baseline(
            "BASELINE B (SIFT + BoVW + SVM)",
            bovw_descriptor, bovw_pipeline,
            train_images, train_labels,
            test_images, test_labels,
            class_names, unlabeled_subset
        )
        if result_b:
            all_results.append(result_b)
    except Exception as e:
        print(f"❌ Failed to evaluate Baseline B: {str(e)}")
    
    # Evaluate Baseline C: Color + LBP + k-NN
    try:
        color_desc, lbp_desc, knn_pipeline = create_baseline_c()
        result_c = evaluate_baseline_c_combined(
            color_desc, lbp_desc, knn_pipeline,
            train_images, train_labels,
            test_images, test_labels,
            class_names
        )
        if result_c:
            all_results.append(result_c)
    except Exception as e:
        print(f"❌ Failed to evaluate Baseline C: {str(e)}")
    
    # Summary
    print(f"\n{'='*60}")
    print("🏆 BASELINE COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    if all_results:
        print("\n📊 Results Summary:")
        print(f"{'Baseline':<35} {'Test Acc':<10} {'Dims':<8} {'Time/img':<12}")
        print("-" * 70)
        
        best_accuracy = 0
        best_baseline = None
        
        for result in all_results:
            acc = result['test_accuracy']
            if acc > best_accuracy:
                best_accuracy = acc
                best_baseline = result['name']
            
            print(f"{result['name']:<35} {acc:<10.4f} {result['feature_dimensions']:<8} "
                  f"{result['avg_time_per_image']*1000:<8.2f} ms")
        
        print(f"\n🥇 Best performing baseline: {best_baseline}")
        print(f"   📊 Test accuracy: {best_accuracy:.4f}")
        
        # Quick efficiency comparison
        fastest_result = min(all_results, key=lambda x: x['avg_time_per_image'])
        print(f"\n⚡ Fastest baseline: {fastest_result['name']}")
        print(f"   ⏱️  Time per image: {fastest_result['avg_time_per_image']*1000:.2f} ms")
    
    else:
        print("❌ No baselines completed successfully")
    
    print(f"\n🎉 Baseline evaluation completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n💡 To run the full evaluation with all descriptors:")
    print("   python scripts/run_full_evaluation.py")


if __name__ == "__main__":
    main()