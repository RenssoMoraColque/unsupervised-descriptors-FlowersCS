"""
Image Preprocessing Utilities
============================

Utilities for image preprocessing and descriptor post-processing.
"""

import numpy as np
from typing import List, Union, Tuple, Optional, Dict, Any
from PIL import Image, ImageEnhance, ImageFilter
import cv2
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
import warnings


class ImagePreprocessor:
    """
    Image preprocessing utilities for descriptor extraction.
    """
    
    def __init__(self, target_size: Optional[Tuple[int, int]] = None):
        """
        Initialize image preprocessor.
        
        Parameters:
        -----------
        target_size : tuple, optional
            Target size (width, height) for resizing images
        """
        self.target_size = target_size
    
    def preprocess_image(self, 
                        image: Union[np.ndarray, Image.Image],
                        grayscale: bool = False,
                        normalize: bool = True,
                        resize: bool = True) -> Union[np.ndarray, Image.Image]:
        """
        Preprocess a single image.
        
        Parameters:
        -----------
        image : np.ndarray or PIL.Image
            Input image
        grayscale : bool
            Convert to grayscale
        normalize : bool
            Normalize pixel values to [0, 1]
        resize : bool
            Resize to target size
            
        Returns:
        --------
        processed_image : same type as input
            Preprocessed image
        """
        # Convert to PIL if numpy array
        was_numpy = isinstance(image, np.ndarray)
        if was_numpy:
            if image.dtype == np.float32 or image.dtype == np.float64:
                image = Image.fromarray((image * 255).astype(np.uint8))
            else:
                image = Image.fromarray(image)
        
        # Resize if requested and target size is set
        if resize and self.target_size is not None:
            image = image.resize(self.target_size, Image.LANCZOS)
        
        # Convert to grayscale if requested
        if grayscale and image.mode != 'L':
            image = image.convert('L')
        
        # Convert back to numpy if original was numpy
        if was_numpy:
            processed = np.array(image)
            if normalize and processed.dtype == np.uint8:
                processed = processed.astype(np.float32) / 255.0
            return processed
        else:
            return image
    
    def preprocess_batch(self, 
                        images: List[Union[np.ndarray, Image.Image]],
                        **kwargs) -> List[Union[np.ndarray, Image.Image]]:
        """
        Preprocess a batch of images.
        
        Parameters:
        -----------
        images : List
            List of input images
        **kwargs : dict
            Arguments passed to preprocess_image
            
        Returns:
        --------
        processed_images : List
            List of preprocessed images
        """
        return [self.preprocess_image(img, **kwargs) for img in images]
    
    def augment_image(self, 
                     image: Union[np.ndarray, Image.Image],
                     brightness_range: Tuple[float, float] = (0.8, 1.2),
                     contrast_range: Tuple[float, float] = (0.8, 1.2),
                     rotation_range: Tuple[float, float] = (-10, 10),
                     apply_blur: bool = False,
                     blur_sigma: float = 1.0) -> Union[np.ndarray, Image.Image]:
        """
        Apply random augmentations to an image.
        
        Parameters:
        -----------
        image : np.ndarray or PIL.Image
            Input image
        brightness_range : tuple
            Range for brightness adjustment
        contrast_range : tuple
            Range for contrast adjustment
        rotation_range : tuple
            Range for rotation in degrees
        apply_blur : bool
            Whether to apply Gaussian blur
        blur_sigma : float
            Standard deviation for Gaussian blur
            
        Returns:
        --------
        augmented_image : same type as input
            Augmented image
        """
        # Convert to PIL for augmentation
        was_numpy = isinstance(image, np.ndarray)
        if was_numpy:
            if image.dtype == np.float32 or image.dtype == np.float64:
                pil_image = Image.fromarray((image * 255).astype(np.uint8))
            else:
                pil_image = Image.fromarray(image)
        else:
            pil_image = image.copy()
        
        # Random brightness
        brightness_factor = np.random.uniform(*brightness_range)
        enhancer = ImageEnhance.Brightness(pil_image)
        pil_image = enhancer.enhance(brightness_factor)
        
        # Random contrast
        contrast_factor = np.random.uniform(*contrast_range)
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(contrast_factor)
        
        # Random rotation
        if rotation_range[0] != rotation_range[1]:
            rotation_angle = np.random.uniform(*rotation_range)
            pil_image = pil_image.rotate(rotation_angle, expand=True, fillcolor=128)
        
        # Random blur
        if apply_blur:
            pil_image = pil_image.filter(ImageFilter.GaussianBlur(radius=blur_sigma))
        
        # Convert back to original format
        if was_numpy:
            result = np.array(pil_image)
            if image.dtype == np.float32 or image.dtype == np.float64:
                result = result.astype(np.float32) / 255.0
            return result
        else:
            return pil_image
    
    def create_image_pyramid(self, 
                           image: Union[np.ndarray, Image.Image],
                           levels: int = 3,
                           scale_factor: float = 0.5) -> List[Union[np.ndarray, Image.Image]]:
        """
        Create image pyramid for multi-scale analysis.
        
        Parameters:
        -----------
        image : np.ndarray or PIL.Image
            Input image
        levels : int
            Number of pyramid levels
        scale_factor : float
            Scale factor between levels
            
        Returns:
        --------
        pyramid : List
            List of images at different scales
        """
        pyramid = [image]
        current_image = image
        
        for _ in range(levels - 1):
            if isinstance(current_image, np.ndarray):
                h, w = current_image.shape[:2]
                new_h, new_w = int(h * scale_factor), int(w * scale_factor)
                current_image = cv2.resize(current_image, (new_w, new_h), 
                                         interpolation=cv2.INTER_LANCZOS4)
            else:
                w, h = current_image.size
                new_w, new_h = int(w * scale_factor), int(h * scale_factor)
                current_image = current_image.resize((new_w, new_h), Image.LANCZOS)
            
            pyramid.append(current_image)
        
        return pyramid


class DescriptorPostprocessor:
    """
    Post-processing utilities for descriptor features.
    """
    
    def __init__(self):
        """Initialize descriptor post-processor."""
        self.scalers = {}
        self.reducers = {}
    
    def normalize_features(self, 
                          features: np.ndarray,
                          method: str = 'standard',
                          fit_transform: bool = True,
                          scaler_key: str = 'default') -> np.ndarray:
        """
        Normalize feature vectors.
        
        Parameters:
        -----------
        features : np.ndarray
            Feature matrix, shape (n_samples, n_features)
        method : str
            Normalization method ('standard', 'minmax', 'robust', 'l2')
        fit_transform : bool
            Whether to fit and transform (True) or just transform (False)
        scaler_key : str
            Key to store/retrieve scaler
            
        Returns:
        --------
        normalized_features : np.ndarray
            Normalized feature matrix
        """
        if method == 'l2':
            # L2 normalization (per sample)
            from sklearn.preprocessing import normalize
            return normalize(features, norm='l2', axis=1)
        
        # Create scaler if needed
        if fit_transform or scaler_key not in self.scalers:
            if method == 'standard':
                scaler = StandardScaler()
            elif method == 'minmax':
                scaler = MinMaxScaler()
            elif method == 'robust':
                scaler = RobustScaler()
            else:
                raise ValueError(f"Unknown normalization method: {method}")
            
            if fit_transform:
                normalized = scaler.fit_transform(features)
                self.scalers[scaler_key] = scaler
            else:
                # This shouldn't happen with fit_transform=False and new scaler
                normalized = features
        else:
            # Use existing scaler
            scaler = self.scalers[scaler_key]
            normalized = scaler.transform(features)
        
        return normalized.astype(np.float32)
    
    def reduce_dimensionality(self,
                            features: np.ndarray,
                            method: str = 'pca',
                            n_components: int = 512,
                            fit_transform: bool = True,
                            reducer_key: str = 'default') -> np.ndarray:
        """
        Reduce dimensionality of features.
        
        Parameters:
        -----------
        features : np.ndarray
            Feature matrix, shape (n_samples, n_features)
        method : str
            Reduction method ('pca', 'ica', 'random_projection')
        n_components : int
            Number of components to keep
        fit_transform : bool
            Whether to fit and transform or just transform
        reducer_key : str
            Key to store/retrieve reducer
            
        Returns:
        --------
        reduced_features : np.ndarray
            Reduced feature matrix
        """
        # Ensure n_components doesn't exceed feature dimensions
        n_components = min(n_components, features.shape[1], features.shape[0])
        
        # Create reducer if needed
        if fit_transform or reducer_key not in self.reducers:
            if method == 'pca':
                reducer = PCA(n_components=n_components, random_state=42)
            elif method == 'ica':
                reducer = FastICA(n_components=n_components, random_state=42, max_iter=1000)
            elif method == 'random_projection':
                reducer = GaussianRandomProjection(n_components=n_components, random_state=42)
            else:
                raise ValueError(f"Unknown reduction method: {method}")
            
            if fit_transform:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    reduced = reducer.fit_transform(features)
                self.reducers[reducer_key] = reducer
            else:
                reduced = features
        else:
            # Use existing reducer
            reducer = self.reducers[reducer_key]
            reduced = reducer.transform(features)
        
        return reduced.astype(np.float32)
    
    def concatenate_descriptors(self, *descriptor_arrays: np.ndarray) -> np.ndarray:
        """
        Concatenate multiple descriptor arrays.
        
        Parameters:
        -----------
        *descriptor_arrays : np.ndarray
            Variable number of descriptor arrays
            
        Returns:
        --------
        concatenated : np.ndarray
            Concatenated descriptor array
        """
        # Check that all arrays have same number of samples
        n_samples = descriptor_arrays[0].shape[0]
        for arr in descriptor_arrays[1:]:
            if arr.shape[0] != n_samples:
                raise ValueError("All descriptor arrays must have same number of samples")
        
        return np.concatenate(descriptor_arrays, axis=1)
    
    def apply_power_normalization(self, features: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """
        Apply power normalization (signed square root).
        
        Parameters:
        -----------
        features : np.ndarray
            Feature matrix
        alpha : float
            Power parameter (0.5 for square root)
            
        Returns:
        --------
        power_normalized : np.ndarray
            Power normalized features
        """
        return np.sign(features) * np.power(np.abs(features), alpha)
    
    def whiten_features(self, features: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
        """
        Apply whitening transformation to features.
        
        Parameters:
        -----------
        features : np.ndarray
            Feature matrix
        epsilon : float
            Regularization parameter
            
        Returns:
        --------
        whitened : np.ndarray
            Whitened features
        """
        # Center the data
        mean = np.mean(features, axis=0)
        centered = features - mean
        
        # Compute covariance matrix
        cov = np.cov(centered.T)
        
        # Eigendecomposition
        eigenvals, eigenvecs = np.linalg.eigh(cov)
        
        # Whitening transformation
        whitening_matrix = eigenvecs @ np.diag(1.0 / np.sqrt(eigenvals + epsilon)) @ eigenvecs.T
        
        return centered @ whitening_matrix.T
    
    def apply_pca(self, features: np.ndarray, n_components: int = None, **kwargs) -> np.ndarray:
        """
        Apply PCA dimensionality reduction.
        
        Convenience method that wraps reduce_dimensionality with PCA.
        
        Parameters:
        -----------
        features : np.ndarray
            Feature matrix, shape (n_samples, n_features)
        n_components : int
            Number of PCA components to keep
        **kwargs : dict
            Additional arguments passed to reduce_dimensionality
            
        Returns:
        --------
        reduced_features : np.ndarray
            PCA-reduced feature matrix
        """
        if n_components is None:
            n_components = min(features.shape[1], features.shape[0])
        
        return self.reduce_dimensionality(
            features, 
            method='pca', 
            n_components=n_components,
            **kwargs
        )
    
    def remove_outliers(self, 
                       features: np.ndarray,
                       method: str = 'zscore',
                       threshold: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove outlier samples from features.
        
        Parameters:
        -----------
        features : np.ndarray
            Feature matrix
        method : str
            Outlier detection method ('zscore', 'iqr')
        threshold : float
            Threshold for outlier detection
            
        Returns:
        --------
        clean_features : np.ndarray
            Features with outliers removed
        mask : np.ndarray
            Boolean mask indicating which samples were kept
        """
        if method == 'zscore':
            # Z-score based outlier detection
            z_scores = np.abs((features - np.mean(features, axis=0)) / np.std(features, axis=0))
            mask = np.all(z_scores < threshold, axis=1)
        elif method == 'iqr':
            # IQR based outlier detection
            Q1 = np.percentile(features, 25, axis=0)
            Q3 = np.percentile(features, 75, axis=0)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            mask = np.all((features >= lower_bound) & (features <= upper_bound), axis=1)
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
        
        return features[mask], mask
    
    def save_preprocessor_state(self, filepath: str) -> None:
        """Save the current state of scalers and reducers."""
        import joblib
        state = {
            'scalers': self.scalers,
            'reducers': self.reducers
        }
        joblib.dump(state, filepath)
    
    def load_preprocessor_state(self, filepath: str) -> None:
        """Load the state of scalers and reducers."""
        import joblib
        state = joblib.load(filepath)
        self.scalers = state.get('scalers', {})
        self.reducers = state.get('reducers', {})