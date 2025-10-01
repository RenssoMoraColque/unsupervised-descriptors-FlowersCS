"""
Encoding Methods for Local Descriptors
======================================

Implementations of encoding methods to convert variable-length local
descriptors into fixed-length global representations.
"""

import numpy as np
from typing import Union, List, Optional, Tuple
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import warnings

from .base import BaseDescriptor


class BagOfVisualWords(BaseDescriptor):
    """
    Bag of Visual Words (BoVW) encoding.
    
    Converts local descriptors to fixed-length histograms using
    k-means clustering to create a visual vocabulary.
    """
    
    def __init__(self, 
                 local_descriptor,
                 codebook_size=512,
                 max_descriptors_per_image=100,
                 random_state=42,
                 **kwargs):
        """
        Initialize BoVW encoder.
        
        Parameters:
        -----------
        local_descriptor : BaseDescriptor
            Local descriptor extractor (SIFT, ORB, etc.)
        codebook_size : int
            Size of visual vocabulary (number of clusters)
        max_descriptors_per_image : int
            Maximum local descriptors to sample per image
        random_state : int
            Random seed for k-means
        """
        super().__init__(
            codebook_size=codebook_size,
            max_descriptors_per_image=max_descriptors_per_image,
            random_state=random_state,
            **kwargs
        )
        self.local_descriptor = local_descriptor
        self.kmeans = None
        self.scaler = StandardScaler()
        
    def fit(self, images: List[Union[np.ndarray, Image.Image]], **kwargs) -> 'BagOfVisualWords':
        """
        Fit BoVW encoder by creating visual vocabulary.
        
        Parameters:
        -----------
        images : List of images
            Unlabeled images for building vocabulary
        """
        print("Fitting local descriptor...")
        self.local_descriptor.fit(images)
        
        print("Extracting local descriptors for vocabulary...")
        all_descriptors = []
        
        for i, img in enumerate(images):
            if i % 100 == 0:
                print(f"Processing image {i}/{len(images)}")
                
            _, descriptors = self.local_descriptor.extract(img)
            
            if len(descriptors) > 0:
                # Sample descriptors if too many
                if len(descriptors) > self.params['max_descriptors_per_image']:
                    indices = np.random.choice(
                        len(descriptors), 
                        self.params['max_descriptors_per_image'],
                        replace=False
                    )
                    descriptors = descriptors[indices]
                
                all_descriptors.append(descriptors)
        
        if not all_descriptors:
            raise ValueError("No descriptors found in any image!")
            
        # Concatenate all descriptors
        all_descriptors = np.vstack(all_descriptors)
        print(f"Total descriptors for clustering: {len(all_descriptors)}")
        
        # Normalize descriptors
        all_descriptors = self.scaler.fit_transform(all_descriptors)
        
        # Create visual vocabulary using k-means
        print(f"Creating vocabulary with {self.params['codebook_size']} words...")
        self.kmeans = KMeans(
            n_clusters=self.params['codebook_size'],
            random_state=self.params['random_state'],
            n_init=10,
            max_iter=300
        )
        self.kmeans.fit(all_descriptors)
        
        self._descriptor_dim = self.params['codebook_size']
        self.is_trained = True
        print("BoVW training completed!")
        return self
        
    def extract(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """
        Extract BoVW histogram from single image.
        
        Returns:
        --------
        histogram : np.ndarray
            BoVW histogram of length codebook_size
        """
        if not self.is_trained:
            raise ValueError("BoVW not trained. Call fit() first.")
            
        _, descriptors = self.local_descriptor.extract(image)
        
        if len(descriptors) == 0:
            # No descriptors found, return zero histogram
            return np.zeros(self.params['codebook_size'])
            
        # Normalize descriptors
        descriptors = self.scaler.transform(descriptors)
        
        # Assign descriptors to clusters
        cluster_assignments = self.kmeans.predict(descriptors)
        
        # Create histogram
        histogram = np.bincount(
            cluster_assignments, 
            minlength=self.params['codebook_size']
        )
        
        # Normalize histogram (L2 normalization)
        histogram = histogram.astype(np.float32)
        norm = np.linalg.norm(histogram)
        if norm > 0:
            histogram = histogram / norm
            
        return histogram


class VLADEncoder(BaseDescriptor):
    """
    Vector of Locally Aggregated Descriptors (VLAD) encoding.
    
    Encodes local descriptors by computing residuals from cluster centers
    and aggregating them into a fixed-length representation.
    """
    
    def __init__(self,
                 local_descriptor,
                 codebook_size=256,
                 max_descriptors_per_image=100,
                 power_normalization=True,
                 l2_normalization=True,
                 random_state=42,
                 **kwargs):
        """
        Initialize VLAD encoder.
        
        Parameters:
        -----------
        local_descriptor : BaseDescriptor
            Local descriptor extractor
        codebook_size : int
            Number of cluster centers
        max_descriptors_per_image : int
            Maximum descriptors to sample per image
        power_normalization : bool
            Apply power normalization (signed square root)
        l2_normalization : bool
            Apply L2 normalization
        random_state : int
            Random seed
        """
        super().__init__(
            codebook_size=codebook_size,
            max_descriptors_per_image=max_descriptors_per_image,
            power_normalization=power_normalization,
            l2_normalization=l2_normalization,
            random_state=random_state,
            **kwargs
        )
        self.local_descriptor = local_descriptor
        self.kmeans = None
        self.scaler = StandardScaler()
        
    def fit(self, images: List[Union[np.ndarray, Image.Image]], **kwargs) -> 'VLADEncoder':
        """Fit VLAD encoder by creating cluster centers."""
        print("Fitting local descriptor...")
        self.local_descriptor.fit(images)
        
        print("Extracting local descriptors for VLAD...")
        all_descriptors = []
        
        for i, img in enumerate(images):
            if i % 100 == 0:
                print(f"Processing image {i}/{len(images)}")
                
            _, descriptors = self.local_descriptor.extract(img)
            
            if len(descriptors) > 0:
                if len(descriptors) > self.params['max_descriptors_per_image']:
                    indices = np.random.choice(
                        len(descriptors),
                        self.params['max_descriptors_per_image'],
                        replace=False
                    )
                    descriptors = descriptors[indices]
                
                all_descriptors.append(descriptors)
        
        if not all_descriptors:
            raise ValueError("No descriptors found in any image!")
            
        all_descriptors = np.vstack(all_descriptors)
        print(f"Total descriptors for clustering: {len(all_descriptors)}")
        
        # Normalize descriptors
        all_descriptors = self.scaler.fit_transform(all_descriptors)
        
        # Create cluster centers
        print(f"Creating {self.params['codebook_size']} cluster centers...")
        self.kmeans = KMeans(
            n_clusters=self.params['codebook_size'],
            random_state=self.params['random_state'],
            n_init=10,
            max_iter=300
        )
        self.kmeans.fit(all_descriptors)
        
        # VLAD dimension = codebook_size * descriptor_dimension
        descriptor_dim = all_descriptors.shape[1]
        self._descriptor_dim = self.params['codebook_size'] * descriptor_dim
        self.is_trained = True
        print("VLAD training completed!")
        return self
        
    def extract(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """Extract VLAD representation from single image."""
        if not self.is_trained:
            raise ValueError("VLAD not trained. Call fit() first.")
            
        _, descriptors = self.local_descriptor.extract(image)
        
        if len(descriptors) == 0:
            return np.zeros(self._descriptor_dim)
            
        # Normalize descriptors
        descriptors = self.scaler.transform(descriptors)
        
        # Get cluster assignments and centers
        cluster_assignments = self.kmeans.predict(descriptors)
        centers = self.kmeans.cluster_centers_
        
        # Compute VLAD representation
        vlad = np.zeros((self.params['codebook_size'], descriptors.shape[1]))
        
        for i in range(self.params['codebook_size']):
            # Find descriptors assigned to cluster i
            cluster_mask = cluster_assignments == i
            
            if np.any(cluster_mask):
                # Compute residuals
                residuals = descriptors[cluster_mask] - centers[i]
                # Sum residuals
                vlad[i] = np.sum(residuals, axis=0)
        
        # Flatten VLAD representation
        vlad = vlad.flatten()
        
        # Apply normalizations
        if self.params['power_normalization']:
            vlad = np.sign(vlad) * np.sqrt(np.abs(vlad))
            
        if self.params['l2_normalization']:
            norm = np.linalg.norm(vlad)
            if norm > 0:
                vlad = vlad / norm
                
        return vlad


class FisherVectorEncoder(BaseDescriptor):
    """
    Fisher Vector encoding.
    
    Encodes local descriptors using Fisher kernel with Gaussian Mixture Model.
    Simplified implementation focusing on first and second order statistics.
    """
    
    def __init__(self,
                 local_descriptor,
                 n_components=128,
                 max_descriptors_per_image=100,
                 power_normalization=True,
                 l2_normalization=True,
                 random_state=42,
                 **kwargs):
        """
        Initialize Fisher Vector encoder.
        
        Parameters:
        -----------
        local_descriptor : BaseDescriptor
            Local descriptor extractor
        n_components : int
            Number of Gaussian components
        max_descriptors_per_image : int
            Maximum descriptors to sample per image
        power_normalization : bool
            Apply power normalization
        l2_normalization : bool
            Apply L2 normalization
        random_state : int
            Random seed
        """
        super().__init__(
            n_components=n_components,
            max_descriptors_per_image=max_descriptors_per_image,
            power_normalization=power_normalization,
            l2_normalization=l2_normalization,
            random_state=random_state,
            **kwargs
        )
        self.local_descriptor = local_descriptor
        self.gmm = None
        self.scaler = StandardScaler()
        
    def fit(self, images: List[Union[np.ndarray, Image.Image]], **kwargs) -> 'FisherVectorEncoder':
        """Fit Fisher Vector encoder by training GMM."""
        print("Fitting local descriptor...")
        self.local_descriptor.fit(images)
        
        print("Extracting local descriptors for Fisher Vector...")
        all_descriptors = []
        
        for i, img in enumerate(images):
            if i % 100 == 0:
                print(f"Processing image {i}/{len(images)}")
                
            _, descriptors = self.local_descriptor.extract(img)
            
            if len(descriptors) > 0:
                if len(descriptors) > self.params['max_descriptors_per_image']:
                    indices = np.random.choice(
                        len(descriptors),
                        self.params['max_descriptors_per_image'],
                        replace=False
                    )
                    descriptors = descriptors[indices]
                
                all_descriptors.append(descriptors)
        
        if not all_descriptors:
            raise ValueError("No descriptors found in any image!")
            
        all_descriptors = np.vstack(all_descriptors)
        print(f"Total descriptors for GMM: {len(all_descriptors)}")
        
        # Normalize descriptors
        all_descriptors = self.scaler.fit_transform(all_descriptors)
        
        # Train GMM
        print(f"Training GMM with {self.params['n_components']} components...")
        self.gmm = GaussianMixture(
            n_components=self.params['n_components'],
            random_state=self.params['random_state'],
            max_iter=100,
            covariance_type='diag'  # Diagonal covariance for efficiency
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.gmm.fit(all_descriptors)
        
        # Fisher Vector dimension = 2 * n_components * descriptor_dimension
        descriptor_dim = all_descriptors.shape[1]
        self._descriptor_dim = 2 * self.params['n_components'] * descriptor_dim
        self.is_trained = True
        print("Fisher Vector training completed!")
        return self
        
    def extract(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """Extract Fisher Vector from single image."""
        if not self.is_trained:
            raise ValueError("Fisher Vector not trained. Call fit() first.")
            
        _, descriptors = self.local_descriptor.extract(image)
        
        if len(descriptors) == 0:
            return np.zeros(self._descriptor_dim)
            
        # Normalize descriptors
        descriptors = self.scaler.transform(descriptors)
        
        # Compute Fisher Vector
        fv = self._compute_fisher_vector(descriptors)
        
        # Apply normalizations
        if self.params['power_normalization']:
            fv = np.sign(fv) * np.sqrt(np.abs(fv))
            
        if self.params['l2_normalization']:
            norm = np.linalg.norm(fv)
            if norm > 0:
                fv = fv / norm
                
        return fv
        
    def _compute_fisher_vector(self, descriptors: np.ndarray) -> np.ndarray:
        """Compute Fisher Vector for given descriptors."""
        # Get GMM parameters
        weights = self.gmm.weights_
        means = self.gmm.means_
        covariances = self.gmm.covariances_
        
        # Compute posterior probabilities
        posteriors = self.gmm.predict_proba(descriptors)  # Shape: (n_desc, n_components)
        
        # Initialize Fisher Vector
        n_components = self.params['n_components']
        descriptor_dim = descriptors.shape[1]
        fv = np.zeros(2 * n_components * descriptor_dim)
        
        # Compute first and second order statistics
        for k in range(n_components):
            # First order statistic (mean deviation)
            diff = descriptors - means[k]
            first_order = np.sum(posteriors[:, k][:, np.newaxis] * diff, axis=0)
            first_order = first_order / (np.sqrt(weights[k]) * np.sqrt(covariances[k]))
            
            # Second order statistic (variance deviation)
            second_order = np.sum(
                posteriors[:, k][:, np.newaxis] * (diff**2 / covariances[k] - 1),
                axis=0
            )
            second_order = second_order / (np.sqrt(2 * weights[k]))
            
            # Store in Fisher Vector
            start_idx = k * descriptor_dim
            end_idx = (k + 1) * descriptor_dim
            fv[start_idx:end_idx] = first_order
            fv[n_components * descriptor_dim + start_idx:n_components * descriptor_dim + end_idx] = second_order
        
        return fv