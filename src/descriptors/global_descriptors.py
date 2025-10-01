"""
Global Image Descriptors
========================

Implementations of global image descriptors that extract a single
feature vector from the entire image.
"""

import numpy as np
from typing import Union, List, Optional
from PIL import Image
import cv2
from skimage.feature import hog, local_binary_pattern
from skimage.color import rgb2gray
from sklearn.preprocessing import StandardScaler

from .base import BaseDescriptor


class HOGDescriptor(BaseDescriptor):
    """
    Histogram of Oriented Gradients (HOG) descriptor.
    
    Extracts gradient orientation histograms from image patches.
    """
    
    def __init__(self, 
                 pixels_per_cell=(8, 8),
                 cells_per_block=(2, 2),
                 orientations=9,
                 **kwargs):
        """
        Initialize HOG descriptor.
        
        Parameters:
        -----------
        pixels_per_cell : tuple
            Size of cell in pixels
        cells_per_block : tuple  
            Number of cells per block
        orientations : int
            Number of orientation bins
        """
        super().__init__(
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            orientations=orientations,
            **kwargs
        )
        self.scaler = StandardScaler()
        
    def fit(self, images: List[Union[np.ndarray, Image.Image]], **kwargs) -> 'HOGDescriptor':
        """Fit HOG descriptor (trains normalizer on features)."""
        # Extract HOG features from sample of images for normalization
        sample_features = []
        sample_size = min(1000, len(images))  # Use sample for efficiency
        
        for i in range(0, len(images), len(images) // sample_size):
            img = self._preprocess_image(images[i])
            features = self._extract_hog(img)
            sample_features.append(features)
            
        # Fit scaler
        self.scaler.fit(sample_features)
        self._descriptor_dim = len(sample_features[0])
        self.is_trained = True
        return self
        
    def extract(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """Extract HOG descriptor from single image."""
        if not self.is_trained:
            raise ValueError("Descriptor not trained. Call fit() first.")
            
        img = self._preprocess_image(image)
        features = self._extract_hog(img)
        normalized_features = self.scaler.transform([features])[0]
        return normalized_features
        
    def _preprocess_image(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """Convert image to grayscale numpy array."""
        if isinstance(image, Image.Image):
            image = np.array(image)
            
        if len(image.shape) == 3:
            image = rgb2gray(image)
            
        return image
        
    def _extract_hog(self, image: np.ndarray) -> np.ndarray:
        """Extract HOG features from preprocessed image."""
        features = hog(
            image,
            orientations=self.params['orientations'],
            pixels_per_cell=self.params['pixels_per_cell'],
            cells_per_block=self.params['cells_per_block'],
            feature_vector=True
        )
        return features


class LBPDescriptor(BaseDescriptor):
    """
    Local Binary Pattern (LBP) descriptor.
    
    Extracts texture information using local binary patterns.
    """
    
    def __init__(self, radius=3, n_points=24, method='uniform', **kwargs):
        """
        Initialize LBP descriptor.
        
        Parameters:
        -----------
        radius : int
            Radius of sample points
        n_points : int
            Number of sample points
        method : str
            LBP method ('uniform', 'default', etc.)
        """
        super().__init__(
            radius=radius,
            n_points=n_points,
            method=method,
            **kwargs
        )
        
    def fit(self, images: List[Union[np.ndarray, Image.Image]], **kwargs) -> 'LBPDescriptor':
        """Fit LBP descriptor."""
        # LBP doesn't require training, just set dimension
        sample_img = self._preprocess_image(images[0])
        sample_features = self._extract_lbp(sample_img)
        self._descriptor_dim = len(sample_features)
        self.is_trained = True
        return self
        
    def extract(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """Extract LBP descriptor from single image."""
        if not self.is_trained:
            raise ValueError("Descriptor not trained. Call fit() first.")
            
        img = self._preprocess_image(image)
        features = self._extract_lbp(img)
        return features
        
    def _preprocess_image(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """Convert image to grayscale numpy array."""
        if isinstance(image, Image.Image):
            image = np.array(image)
            
        if len(image.shape) == 3:
            image = rgb2gray(image)
            
        return image
        
    def _extract_lbp(self, image: np.ndarray) -> np.ndarray:
        """Extract LBP histogram from preprocessed image."""
        lbp = local_binary_pattern(
            image,
            P=self.params['n_points'],
            R=self.params['radius'],
            method=self.params['method']
        )
        
        # Create histogram
        n_bins = self.params['n_points'] + 2
        hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
        return hist


class ColorHistogramDescriptor(BaseDescriptor):
    """
    Color Histogram descriptor.
    
    Extracts color distribution histograms in HSV color space.
    """
    
    def __init__(self, h_bins=16, s_bins=16, v_bins=16, **kwargs):
        """
        Initialize Color Histogram descriptor.
        
        Parameters:
        -----------
        h_bins : int
            Number of hue bins
        s_bins : int
            Number of saturation bins  
        v_bins : int
            Number of value bins
        """
        super().__init__(
            h_bins=h_bins,
            s_bins=s_bins,
            v_bins=v_bins,
            **kwargs
        )
        
    def fit(self, images: List[Union[np.ndarray, Image.Image]], **kwargs) -> 'ColorHistogramDescriptor':
        """Fit Color Histogram descriptor."""
        # Color histograms don't require training
        self._descriptor_dim = (
            self.params['h_bins'] + 
            self.params['s_bins'] + 
            self.params['v_bins']
        )
        self.is_trained = True
        return self
        
    def extract(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """Extract color histogram from single image."""
        if not self.is_trained:
            raise ValueError("Descriptor not trained. Call fit() first.")
            
        img = self._preprocess_image(image)
        features = self._extract_color_histogram(img)
        return features
        
    def _preprocess_image(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """Convert image to HSV numpy array."""
        if isinstance(image, Image.Image):
            image = np.array(image)
            
        if len(image.shape) == 3:
            # Convert RGB to HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        else:
            raise ValueError("Color histogram requires color images")
            
        return hsv
        
    def _extract_color_histogram(self, hsv_image: np.ndarray) -> np.ndarray:
        """Extract color histogram from HSV image."""
        h, s, v = cv2.split(hsv_image)
        
        # Compute histograms for each channel
        hist_h = cv2.calcHist([h], [0], None, [self.params['h_bins']], [0, 180])
        hist_s = cv2.calcHist([s], [0], None, [self.params['s_bins']], [0, 256])
        hist_v = cv2.calcHist([v], [0], None, [self.params['v_bins']], [0, 256])
        
        # Normalize and concatenate
        hist_h = hist_h.flatten() / np.sum(hist_h)
        hist_s = hist_s.flatten() / np.sum(hist_s)
        hist_v = hist_v.flatten() / np.sum(hist_v)
        
        return np.concatenate([hist_h, hist_s, hist_v])


class GISTDescriptor(BaseDescriptor):
    """
    GIST descriptor - simplified implementation.
    
    Extracts global scene representation using Gabor filters.
    Note: This is a simplified version. Full GIST implementation 
    would require more complex Gabor filter banks.
    """
    
    def __init__(self, n_blocks=4, n_orientations=4, **kwargs):
        """
        Initialize GIST descriptor.
        
        Parameters:
        -----------
        n_blocks : int
            Number of spatial blocks per dimension
        n_orientations : int
            Number of orientation filters
        """
        super().__init__(
            n_blocks=n_blocks,
            n_orientations=n_orientations,
            **kwargs
        )
        
    def fit(self, images: List[Union[np.ndarray, Image.Image]], **kwargs) -> 'GISTDescriptor':
        """Fit GIST descriptor."""
        # GIST doesn't require training
        self._descriptor_dim = (
            self.params['n_blocks'] ** 2 * 
            self.params['n_orientations']
        )
        self.is_trained = True
        return self
        
    def extract(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """Extract GIST descriptor from single image."""
        if not self.is_trained:
            raise ValueError("Descriptor not trained. Call fit() first.")
            
        img = self._preprocess_image(image)
        features = self._extract_gist(img)
        return features
        
    def _preprocess_image(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """Convert image to grayscale numpy array."""
        if isinstance(image, Image.Image):
            image = np.array(image)
            
        if len(image.shape) == 3:
            image = rgb2gray(image)
            
        return image
        
    def _extract_gist(self, image: np.ndarray) -> np.ndarray:
        """Extract simplified GIST features."""
        # Simplified GIST: use gradient histograms in spatial blocks
        h, w = image.shape
        n_blocks = self.params['n_blocks']
        n_orientations = self.params['n_orientations']
        
        block_h = h // n_blocks
        block_w = w // n_blocks
        
        features = []
        
        for i in range(n_blocks):
            for j in range(n_blocks):
                # Extract block
                start_h = i * block_h
                end_h = min((i + 1) * block_h, h)
                start_w = j * block_w  
                end_w = min((j + 1) * block_w, w)
                
                block = image[start_h:end_h, start_w:end_w]
                
                # Compute gradient orientations
                gx = cv2.Sobel(block, cv2.CV_64F, 1, 0, ksize=3)
                gy = cv2.Sobel(block, cv2.CV_64F, 0, 1, ksize=3)
                
                magnitude = np.sqrt(gx**2 + gy**2)
                orientation = np.arctan2(gy, gx)
                
                # Create orientation histogram
                hist, _ = np.histogram(
                    orientation,
                    bins=n_orientations,
                    range=(-np.pi, np.pi),
                    weights=magnitude
                )
                hist = hist / (np.sum(hist) + 1e-7)  # Normalize
                features.extend(hist)
                
        return np.array(features)