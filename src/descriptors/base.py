"""
Base Descriptor Interface
=========================

Abstract base class for all image descriptors.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Union, List, Any, Optional
from PIL import Image


class BaseDescriptor(ABC):
    """
    Abstract base class for all image descriptors.
    
    All descriptor implementations should inherit from this class and 
    implement the required methods.
    """
    
    def __init__(self, **kwargs):
        """Initialize the descriptor with parameters."""
        self.params = kwargs
        self.is_trained = False
        
    @abstractmethod
    def fit(self, images: List[Union[np.ndarray, Image.Image]], **kwargs) -> 'BaseDescriptor':
        """
        Train/fit the descriptor on unlabeled data.
        
        Parameters:
        -----------
        images : List of images (numpy arrays or PIL Images)
            Unlabeled images for unsupervised training
            
        Returns:
        --------
        self : BaseDescriptor
            Fitted descriptor instance
        """
        pass
    
    @abstractmethod
    def extract(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """
        Extract descriptor from a single image.
        
        Parameters:
        -----------
        image : np.ndarray or PIL.Image
            Input image
            
        Returns:
        --------
        descriptor : np.ndarray
            Fixed-length descriptor vector
        """
        pass
    
    def extract_batch(self, images: List[Union[np.ndarray, Image.Image]]) -> np.ndarray:
        """
        Extract descriptors from a batch of images.
        
        Parameters:
        -----------
        images : List of images
            Input images
            
        Returns:
        --------
        descriptors : np.ndarray
            Array of descriptor vectors, shape (n_images, descriptor_dim)
        """
        descriptors = []
        for img in images:
            desc = self.extract(img)
            descriptors.append(desc)
        return np.array(descriptors)
    
    def get_descriptor_dimension(self) -> int:
        """
        Get the dimensionality of the descriptor.
        
        Returns:
        --------
        dim : int
            Descriptor dimensionality
        """
        if not hasattr(self, '_descriptor_dim'):
            raise ValueError("Descriptor dimension not set. Call fit() first.")
        return self._descriptor_dim
    
    def save(self, filepath: str) -> None:
        """
        Save the trained descriptor to disk.
        
        Parameters:
        -----------
        filepath : str
            Path to save the descriptor
        """
        import joblib
        joblib.dump(self, filepath)
    
    @classmethod
    def load(cls, filepath: str) -> 'BaseDescriptor':
        """
        Load a trained descriptor from disk.
        
        Parameters:
        -----------
        filepath : str
            Path to the saved descriptor
            
        Returns:
        --------
        descriptor : BaseDescriptor
            Loaded descriptor instance
        """
        import joblib
        return joblib.load(filepath)
    
    def __repr__(self) -> str:
        """String representation of the descriptor."""
        class_name = self.__class__.__name__
        params_str = ", ".join([f"{k}={v}" for k, v in self.params.items()])
        return f"{class_name}({params_str})"