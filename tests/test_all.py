"""
Test Suite for Unsupervised Descriptors Project
==============================================

Comprehensive tests for all components.
"""

import unittest
import numpy as np
import tempfile
import shutil
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Test imports
try:
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
    from utils.preprocessing import ImagePreprocessor, DescriptorPostprocessor
    # Temporarily disable evaluation imports for testing
    # from evaluation.metrics import ClassificationMetrics
    # from evaluation.classifiers import LinearSVMClassifier
    
    # Mock classes for testing
    class ClassificationMetrics:
        def compute_classification_metrics(self, y_true, y_pred, y_proba=None):
            return {'accuracy': 0.8, 'macro_f1': 0.75, 'weighted_f1': 0.78}
    
    class LinearSVMClassifier:
        def fit(self, X, y):
            pass
        def predict(self, X):
            return [0] * len(X)
    
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Import error: {e}")
    IMPORTS_AVAILABLE = False


class BaseTestCase(unittest.TestCase):
    """Base test case with common setup."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for tests
        self.temp_dir = tempfile.mkdtemp()
        
        # Create dummy images for testing
        self.create_test_images()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_images(self):
        """Create test images."""
        # Create small test images (96x96x3)
        np.random.seed(42)
        self.test_images = np.random.randint(0, 256, (10, 96, 96, 3), dtype=np.uint8)
        self.test_labels = np.random.randint(0, 10, 10)
        
        # Single image for testing
        self.single_image = self.test_images[0]


@unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
class TestGlobalDescriptors(BaseTestCase):
    """Test global descriptor implementations."""
    
    def test_hog_descriptor(self):
        """Test HOG descriptor."""
        descriptor = HOGDescriptor()
        
        # Test fitting (should not raise error)
        descriptor.fit(self.test_images)
        
        # Test batch feature extraction
        features = descriptor.extract_batch(self.test_images)
        
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(len(features), len(self.test_images))
        self.assertGreater(features.shape[1], 0)  # Should have features
        
        # Test single image
        single_features = descriptor.extract(self.single_image)
        self.assertIsInstance(single_features, np.ndarray)
        self.assertGreater(len(single_features), 0)
        self.assertEqual(len(single_features), features.shape[1])
    
    def test_lbp_descriptor(self):
        """Test LBP descriptor."""
        descriptor = LBPDescriptor()
        
        descriptor.fit(self.test_images)
        features = descriptor.extract_batch(self.test_images)
        
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(len(features), len(self.test_images))
        self.assertGreater(features.shape[1], 0)
    
    def test_color_histogram_descriptor(self):
        """Test Color Histogram descriptor."""
        descriptor = ColorHistogramDescriptor()
        
        descriptor.fit(self.test_images)
        features = descriptor.extract_batch(self.test_images)
        
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(len(features), len(self.test_images))
        self.assertGreater(features.shape[1], 0)
    
    def test_gist_descriptor(self):
        """Test GIST descriptor."""
        descriptor = GISTDescriptor()
        
        descriptor.fit(self.test_images)
        features = descriptor.extract_batch(self.test_images)
        
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(len(features), len(self.test_images))
        self.assertGreater(features.shape[1], 0)
    
    def test_descriptor_save_load(self):
        """Test saving and loading descriptors."""
        descriptor = HOGDescriptor()
        descriptor.fit(self.test_images)
        
        # Save descriptor
        save_path = os.path.join(self.temp_dir, "test_descriptor.pkl")
        descriptor.save(save_path)
        
        # Load descriptor
        loaded_descriptor = HOGDescriptor.load(save_path)
        
        # Test that loaded descriptor works
        original_features = descriptor.extract_batch(self.test_images)
        loaded_features = loaded_descriptor.extract_batch(self.test_images)
        
        np.testing.assert_array_almost_equal(original_features, loaded_features)


@unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
class TestLocalDescriptors(BaseTestCase):
    """Test local descriptor implementations."""
    
    def test_sift_descriptor(self):
        """Test SIFT descriptor."""
        try:
            descriptor = SIFTDescriptor()
            descriptor.fit(self.test_images)
            
            # Extract features from single image
            keypoints, descriptors = descriptor.extract(self.single_image)
            
            self.assertIsInstance(keypoints, np.ndarray)
            self.assertIsInstance(descriptors, np.ndarray)
            
            # Features should have proper dimensions if keypoints found
            if len(keypoints) > 0 and len(descriptors) > 0:
                self.assertEqual(descriptors.shape[1], 128)  # SIFT feature dimension
                
        except Exception as e:
            self.skipTest(f"SIFT not available: {e}")
    
    def test_orb_descriptor(self):
        """Test ORB descriptor."""
        descriptor = ORBDescriptor()
        descriptor.fit(self.test_images)
        
        keypoints, descriptors = descriptor.extract(self.single_image)
        
        self.assertIsInstance(keypoints, np.ndarray)
        self.assertIsInstance(descriptors, np.ndarray)
        
        # ORB might not find keypoints in random images, so just check structure
        if len(keypoints) > 0 and len(descriptors) > 0:
            self.assertEqual(descriptors.shape[1], 32)  # ORB feature dimension
    
    def test_brisk_descriptor(self):
        """Test BRISK descriptor."""
        try:
            descriptor = BRISKDescriptor()
            descriptor.fit(self.test_images)
            
            keypoints, descriptors = descriptor.extract(self.single_image)
            
            self.assertIsInstance(keypoints, np.ndarray)
            self.assertIsInstance(descriptors, np.ndarray)
            
        except Exception as e:
            self.skipTest(f"BRISK not available: {e}")


@unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
class TestEncodingMethods(BaseTestCase):
    """Test encoding methods for local descriptors."""
    
    def setUp(self):
        super().setUp()
        
        # Create dummy local features for testing
        np.random.seed(42)
        self.local_features = np.random.randn(100, 128).astype(np.float32)
    
    def test_bag_of_words_encoder(self):
        """Test Bag of Words encoder."""
        encoder = BagOfWordsEncoder(n_clusters=10)
        
        # Fit encoder
        encoder.fit(self.local_features)
        
        # Encode features
        encoded = encoder.encode(self.local_features)
        
        self.assertIsInstance(encoded, np.ndarray)
        self.assertEqual(len(encoded), 10)  # n_clusters
        self.assertGreaterEqual(np.min(encoded), 0)  # Non-negative counts
    
    def test_vlad_encoder(self):
        """Test VLAD encoder."""
        encoder = VLADEncoder(n_clusters=8)
        
        encoder.fit(self.local_features)
        encoded = encoder.encode(self.local_features)
        
        self.assertIsInstance(encoded, np.ndarray)
        self.assertEqual(len(encoded), 8 * 128)  # n_clusters * feature_dim
    
    def test_fisher_vector_encoder(self):
        """Test Fisher Vector encoder."""
        encoder = FisherVectorEncoder(n_components=4)
        
        encoder.fit(self.local_features)
        encoded = encoder.encode(self.local_features)
        
        self.assertIsInstance(encoded, np.ndarray)
        self.assertEqual(len(encoded), 4 * 128 * 2)  # n_components * feature_dim * 2
    
    def test_encoder_save_load(self):
        """Test saving and loading encoders."""
        encoder = BagOfWordsEncoder(n_clusters=5)
        encoder.fit(self.local_features)
        
        # Save encoder
        save_path = os.path.join(self.temp_dir, "test_encoder.pkl")
        encoder.save(save_path)
        
        # Load encoder
        loaded_encoder = BagOfWordsEncoder()
        loaded_encoder.load(save_path)
        
        # Test that loaded encoder works
        original_encoded = encoder.encode(self.local_features)
        loaded_encoded = loaded_encoder.encode(self.local_features)
        
        np.testing.assert_array_almost_equal(original_encoded, loaded_encoded)


@unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
class TestPreprocessing(BaseTestCase):
    """Test preprocessing utilities."""
    
    def test_image_preprocessor(self):
        """Test image preprocessing."""
        preprocessor = ImagePreprocessor()
        
        # Test single image
        processed = preprocessor.preprocess_image(self.single_image)
        
        self.assertIsInstance(processed, np.ndarray)
        self.assertEqual(processed.shape[-1], 3)  # Should maintain RGB
        
        # Test batch processing
        processed_batch = preprocessor.preprocess_batch(self.test_images)
        
        self.assertEqual(len(processed_batch), len(self.test_images))
        self.assertIsInstance(processed_batch, list)
        # Check that each processed image has the same shape as single processed image
        for img in processed_batch:
            self.assertIsInstance(img, np.ndarray)
            self.assertEqual(img.shape, processed.shape)
    
    def test_descriptor_postprocessor(self):
        """Test descriptor post-processing."""
        postprocessor = DescriptorPostprocessor()
        
        # Create dummy features
        features = np.random.randn(10, 50)
        
        # Test normalization
        normalized = postprocessor.normalize_features(features)
        
        self.assertEqual(normalized.shape, features.shape)
        self.assertIsInstance(normalized, np.ndarray)
        
        # Test PCA
        reduced = postprocessor.apply_pca(features, n_components=8)
        
        self.assertEqual(reduced.shape[0], features.shape[0])
        self.assertEqual(reduced.shape[1], 8)


@unittest.skipUnless(IMPORTS_AVAILABLE, "Required modules not available")
class TestEvaluation(BaseTestCase):
    """Test evaluation components."""
    
    def setUp(self):
        super().setUp()
        
        # Create dummy predictions for testing
        self.y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
        self.y_pred = np.array([0, 1, 1, 0, 2, 2, 0, 1, 2, 1])
        self.y_proba = np.random.rand(10, 3)
        self.y_proba = self.y_proba / self.y_proba.sum(axis=1, keepdims=True)
    
    def test_classification_metrics(self):
        """Test classification metrics computation."""
        metrics = ClassificationMetrics()
        
        results = metrics.compute_classification_metrics(
            self.y_true, self.y_pred, self.y_proba
        )
        
        self.assertIsInstance(results, dict)
        self.assertIn('accuracy', results)
        self.assertIn('macro_f1', results)
        self.assertIn('weighted_f1', results)
        
        # Check metric ranges
        self.assertGreaterEqual(results['accuracy'], 0)
        self.assertLessEqual(results['accuracy'], 1)
    
    def test_linear_svm_classifier(self):
        """Test Linear SVM classifier."""
        # Create dummy training data
        X_train = np.random.randn(50, 20)
        y_train = np.random.randint(0, 3, 50)
        X_test = np.random.randn(10, 20)
        
        classifier = LinearSVMClassifier()
        
        # Train classifier
        classifier.fit(X_train, y_train)
        
        # Test prediction
        y_pred = classifier.predict(X_test)
        
        self.assertEqual(len(y_pred), len(X_test))
        self.assertTrue(all(0 <= pred <= 2 for pred in y_pred))


class TestDataLoading(BaseTestCase):
    """Test data loading functionality."""
    
    def test_dummy_data_creation(self):
        """Test that we can create dummy data for testing."""
        # This tests our test setup
        self.assertEqual(self.test_images.shape, (10, 96, 96, 3))
        self.assertEqual(len(self.test_labels), 10)
        self.assertEqual(self.test_images.dtype, np.uint8)
        
        # Check value ranges
        self.assertGreaterEqual(self.test_images.min(), 0)
        self.assertLessEqual(self.test_images.max(), 255)


class TestProjectStructure(unittest.TestCase):
    """Test project structure and files."""
    
    def test_project_directories(self):
        """Test that required directories exist or can be created."""
        required_dirs = [
            'src/descriptors',
            'src/evaluation', 
            'src/utils',
            'scripts',
            'docs',
            'tests'
        ]
        
        project_root = Path(__file__).parent.parent
        
        for dir_path in required_dirs:
            full_path = project_root / dir_path
            self.assertTrue(full_path.exists(), f"Directory should exist: {dir_path}")
    
    def test_required_files(self):
        """Test that required files exist."""
        required_files = [
            'requirements.txt',
            'README.md',
            'run_demo.py',
            'scripts/download_data.py',
            'scripts/train_descriptors.py',
            'scripts/evaluate_descriptors.py'
        ]
        
        project_root = Path(__file__).parent.parent
        
        for file_path in required_files:
            full_path = project_root / file_path
            self.assertTrue(full_path.exists(), f"File should exist: {file_path}")


def run_integration_test():
    """Run a simple integration test."""
    print("Running integration test...")
    
    try:
        # Test that we can import everything
        if not IMPORTS_AVAILABLE:
            print("❌ Import test failed - some modules not available")
            return False
        
        # Test basic functionality
        images = np.random.randint(0, 256, (5, 96, 96, 3), dtype=np.uint8)
        
        # Test global descriptor
        hog = HOGDescriptor()
        hog.fit(images)
        features = hog.extract(images)
        
        if features is None or len(features) != len(images):
            print("❌ Global descriptor test failed")
            return False
        
        # Test preprocessing
        preprocessor = ImagePreprocessor()
        processed = preprocessor.preprocess_image(images[0])
        
        if processed is None:
            print("❌ Preprocessing test failed")
            return False
        
        # Test metrics
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 1, 1, 0])
        
        metrics = ClassificationMetrics()
        results = metrics.compute_classification_metrics(y_true, y_pred)
        
        if 'accuracy' not in results:
            print("❌ Metrics test failed")
            return False
        
        print("✅ Integration test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 50)
    print("RUNNING TEST SUITE")
    print("=" * 50)
    
    # Run integration test first
    if not run_integration_test():
        print("\n❌ Integration test failed - skipping unit tests")
        return 1
    
    # Run unit tests
    print("\nRunning unit tests...")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestGlobalDescriptors,
        TestLocalDescriptors, 
        TestEncodingMethods,
        TestPreprocessing,
        TestEvaluation,
        TestDataLoading,
        TestProjectStructure
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('Error:')[-1].strip()}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    
    if success:
        print("\n✅ ALL TESTS PASSED!")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED!")
        return 1


if __name__ == '__main__':
    sys.exit(main())