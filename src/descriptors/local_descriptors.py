"""
Local Image Descriptors
=======================

Implementations of local feature detectors and descriptors that extract
keypoints and their descriptions from images.
"""

import numpy as np
from typing import Union, List, Tuple, Optional
from PIL import Image
import cv2
from sklearn.preprocessing import StandardScaler

from .base import BaseDescriptor


class SIFTDescriptor(BaseDescriptor):
    """
    Scale-Invariant Feature Transform (SIFT) descriptor.
    
    Extracts SIFT keypoints and descriptors from images.
    """
    
    def __init__(self, 
                 nfeatures=0,
                 nOctaveLayers=3,
                 contrastThreshold=0.04,
                 edgeThreshold=10,
                 sigma=1.6,
                 **kwargs):
        """
        Initialize SIFT descriptor.
        
        Parameters:
        -----------
        nfeatures : int
            Number of best features to retain (0 = no limit)
        nOctaveLayers : int
            Number of layers in each octave
        contrastThreshold : float
            Contrast threshold for filtering weak features
        edgeThreshold : float
            Edge threshold for filtering edge-like features
        sigma : float
            Gaussian blur sigma for base image
        """
        super().__init__(
            nfeatures=nfeatures,
            nOctaveLayers=nOctaveLayers,
            contrastThreshold=contrastThreshold,
            edgeThreshold=edgeThreshold,
            sigma=sigma,
            **kwargs
        )
        self.sift = cv2.SIFT_create(
            nfeatures=nfeatures,
            nOctaveLayers=nOctaveLayers,
            contrastThreshold=contrastThreshold,
            edgeThreshold=edgeThreshold,
            sigma=sigma
        )
        
    def fit(self, images: List[Union[np.ndarray, Image.Image]], **kwargs) -> 'SIFTDescriptor':
        """Fit SIFT descriptor (no training needed for SIFT)."""
        # SIFT descriptors are 128-dimensional
        self._descriptor_dim = 128
        self.is_trained = True
        return self
        
    def extract(self, image: Union[np.ndarray, Image.Image]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract SIFT keypoints and descriptors from single image.
        
        Returns:
        --------
        keypoints : np.ndarray
            Array of keypoint coordinates, shape (n_keypoints, 2)
        descriptors : np.ndarray
            Array of SIFT descriptors, shape (n_keypoints, 128)
        """
        if not self.is_trained:
            raise ValueError("Descriptor not trained. Call fit() first.")
            
        img = self._preprocess_image(image)
        keypoints, descriptors = self.sift.detectAndCompute(img, None)
        
        if descriptors is None:
            # No keypoints found
            return np.array([]), np.array([])
            
        # Extract keypoint coordinates
        kp_coords = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints])
        
        return kp_coords, descriptors
        
    def _preprocess_image(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """Convert image to grayscale numpy array."""
        if isinstance(image, Image.Image):
            image = np.array(image)
            
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
        return image


class SURFDescriptor(BaseDescriptor):
    """
    Speeded-Up Robust Features (SURF) descriptor.
    
    Note: SURF is patented and not available in OpenCV by default.
    This is a placeholder implementation.
    """
    
    def __init__(self, 
                 hessianThreshold=400,
                 nOctaves=4,
                 nOctaveLayers=3,
                 **kwargs):
        """
        Initialize SURF descriptor.
        
        Parameters:
        -----------
        hessianThreshold : float
            Hessian threshold for keypoint detection
        nOctaves : int
            Number of pyramid octaves
        nOctaveLayers : int
            Number of layers per octave
        """
        super().__init__(
            hessianThreshold=hessianThreshold,
            nOctaves=nOctaves,
            nOctaveLayers=nOctaveLayers,
            **kwargs
        )
        # Note: SURF requires opencv-contrib-python
        try:
            self.surf = cv2.xfeatures2d.SURF_create(hessianThreshold)
        except AttributeError:
            raise ImportError("SURF requires opencv-contrib-python. Use SIFT instead.")
        
    def fit(self, images: List[Union[np.ndarray, Image.Image]], **kwargs) -> 'SURFDescriptor':
        """Fit SURF descriptor (no training needed)."""
        # SURF descriptors are 64 or 128-dimensional
        self._descriptor_dim = 64  # Default SURF size
        self.is_trained = True
        return self
        
    def extract(self, image: Union[np.ndarray, Image.Image]) -> Tuple[np.ndarray, np.ndarray]:
        """Extract SURF keypoints and descriptors from single image."""
        if not self.is_trained:
            raise ValueError("Descriptor not trained. Call fit() first.")
            
        img = self._preprocess_image(image)
        keypoints, descriptors = self.surf.detectAndCompute(img, None)
        
        if descriptors is None:
            return np.array([]), np.array([])
            
        kp_coords = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints])
        return kp_coords, descriptors
        
    def _preprocess_image(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """Convert image to grayscale numpy array."""
        if isinstance(image, Image.Image):
            image = np.array(image)
            
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
        return image


class ORBDescriptor(BaseDescriptor):
    """
    Oriented FAST and Rotated BRIEF (ORB) descriptor.
    
    A fast alternative to SIFT and SURF.
    """
    
    def __init__(self,
                 nfeatures=500,
                 scaleFactor=1.2,
                 nlevels=8,
                 edgeThreshold=31,
                 **kwargs):
        """
        Initialize ORB descriptor.
        
        Parameters:
        -----------
        nfeatures : int
            Maximum number of features to retain
        scaleFactor : float
            Pyramid decimation ratio
        nlevels : int
            Number of pyramid levels
        edgeThreshold : int
            Size of border where features are not detected
        """
        super().__init__(
            nfeatures=nfeatures,
            scaleFactor=scaleFactor,
            nlevels=nlevels,
            edgeThreshold=edgeThreshold,
            **kwargs
        )
        self.orb = cv2.ORB_create(
            nfeatures=nfeatures,
            scaleFactor=scaleFactor,
            nlevels=nlevels,
            edgeThreshold=edgeThreshold
        )
        
    def fit(self, images: List[Union[np.ndarray, Image.Image]], **kwargs) -> 'ORBDescriptor':
        """Fit ORB descriptor (no training needed)."""
        # ORB descriptors are 32-dimensional binary
        self._descriptor_dim = 32
        self.is_trained = True
        return self
        
    def extract(self, image: Union[np.ndarray, Image.Image]) -> Tuple[np.ndarray, np.ndarray]:
        """Extract ORB keypoints and descriptors from single image."""
        if not self.is_trained:
            raise ValueError("Descriptor not trained. Call fit() first.")
            
        img = self._preprocess_image(image)
        keypoints, descriptors = self.orb.detectAndCompute(img, None)
        
        if descriptors is None:
            return np.array([]), np.array([])
            
        kp_coords = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints])
        return kp_coords, descriptors.astype(np.float32)  # Convert binary to float
        
    def _preprocess_image(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """Convert image to grayscale numpy array."""
        if isinstance(image, Image.Image):
            image = np.array(image)
            
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
        return image


class BRISKDescriptor(BaseDescriptor):
    """
    Binary Robust Invariant Scalable Keypoints (BRISK) descriptor.
    
    A binary descriptor similar to ORB but with different keypoint detection.
    """
    
    def __init__(self, thresh=30, octaves=3, patternScale=1.0, **kwargs):
        """
        Initialize BRISK descriptor.
        
        Parameters:
        -----------
        thresh : int
            AGAST detection threshold score
        octaves : int
            Detection octaves
        patternScale : float
            Pattern scale for descriptor computation
        """
        super().__init__(
            thresh=thresh,
            octaves=octaves,
            patternScale=patternScale,
            **kwargs
        )
        self.brisk = cv2.BRISK_create(
            thresh=thresh,
            octaves=octaves,
            patternScale=patternScale
        )
        
    def fit(self, images: List[Union[np.ndarray, Image.Image]], **kwargs) -> 'BRISKDescriptor':
        """Fit BRISK descriptor (no training needed)."""
        # BRISK descriptors are 64-dimensional binary
        self._descriptor_dim = 64
        self.is_trained = True
        return self
        
    def extract(self, image: Union[np.ndarray, Image.Image]) -> Tuple[np.ndarray, np.ndarray]:
        """Extract BRISK keypoints and descriptors from single image."""
        if not self.is_trained:
            raise ValueError("Descriptor not trained. Call fit() first.")
            
        img = self._preprocess_image(image)
        keypoints, descriptors = self.brisk.detectAndCompute(img, None)
        
        if descriptors is None:
            return np.array([]), np.array([])
            
        kp_coords = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints])
        return kp_coords, descriptors.astype(np.float32)  # Convert binary to float
        
    def _preprocess_image(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """Convert image to grayscale numpy array."""
        if isinstance(image, Image.Image):
            image = np.array(image)
            
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
        return image