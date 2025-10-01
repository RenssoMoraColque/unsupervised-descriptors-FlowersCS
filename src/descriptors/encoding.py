"""
Encoding Methods for Local Descriptors
======================================

Implementations of encoding methods to convert variable-length local
descriptors into fixed-length global representations.
"""

import numpy as np
import pickle
from typing import Union, List, Optional, Tuple
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import warnings


class BaseEncoder:
    """Base class for descriptor encoders."""
    
    def __init__(self, **kwargs):
        """Initialize encoder with parameters."""
        self.params = kwargs
        self.is_trained = False
        self._feature_dim = None
    
    def fit(self, features: np.ndarray) -> 'BaseEncoder':
        """Fit encoder on feature set."""
        raise NotImplementedError("Subclasses must implement fit()")
    
    def encode(self, features: np.ndarray) -> np.ndarray:
        """Encode features to fixed-length vector."""
        raise NotImplementedError("Subclasses must implement encode()")
    
    def get_feature_dim(self) -> int:
        """Get dimension of encoded features."""
        return self._feature_dim
    
    def save(self, filepath: str) -> None:
        """Save encoder to file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    def load(self, filepath: str) -> None:
        """Load encoder from file."""
        with open(filepath, 'rb') as f:
            encoder = pickle.load(f)
            self.__dict__.update(encoder.__dict__)


class BagOfWordsEncoder(BaseEncoder):
    """
    Bag of Visual Words (BoVW) encoding.
    
    Converts local descriptors to fixed-length histograms using
    k-means clustering to create a visual vocabulary.
    """
    
    def __init__(self, 
                 n_clusters=512,
                 random_state=42,
                 **kwargs):
        """
        Initialize BoVW encoder.
        
        Parameters:
        -----------
        n_clusters : int
            Size of visual vocabulary (number of clusters)
        random_state : int
            Random seed for k-means
        """
        super().__init__(
            n_clusters=n_clusters,
            random_state=random_state,
            **kwargs
        )
        self.kmeans = None
        self.scaler = StandardScaler()
        
    def fit(self, features: np.ndarray) -> 'BagOfWordsEncoder':
        """
        Fit BoVW encoder by creating visual vocabulary.
        
        Parameters:
        -----------
        features : np.ndarray
            Local features for building vocabulary, shape (n_features, feature_dim)
        """
        if len(features) == 0:
            raise ValueError("No features provided for training!")
            
        print(f"Training BoVW with {len(features)} features...")
        
        # Normalize features
        features = self.scaler.fit_transform(features)
        
        # Create visual vocabulary using k-means
        print(f"Creating vocabulary with {self.params['n_clusters']} words...")
        self.kmeans = KMeans(
            n_clusters=self.params['n_clusters'],
            random_state=self.params['random_state'],
            n_init=10,
            max_iter=300
        )
        self.kmeans.fit(features)
        
        self._feature_dim = self.params['n_clusters']
        self.is_trained = True
        print("BoVW training completed!")
        return self
        
    def encode(self, features: np.ndarray) -> np.ndarray:
        """
        Encode local features to BoVW histogram.
        
        Parameters:
        -----------
        features : np.ndarray
            Local features to encode, shape (n_features, feature_dim)
            
        Returns:
        --------
        histogram : np.ndarray
            BoVW histogram of length n_clusters
        """
        if not self.is_trained:
            raise ValueError("BoVW not trained. Call fit() first.")
            
        if len(features) == 0:
            # No features, return zero histogram
            return np.zeros(self.params['n_clusters'])
            
        # Normalize features
        features = self.scaler.transform(features)
        
        # Assign features to clusters
        cluster_assignments = self.kmeans.predict(features)
        
        # Create histogram
        histogram = np.bincount(
            cluster_assignments, 
            minlength=self.params['n_clusters']
        )
        
        # Normalize histogram (L2 normalization)
        histogram = histogram.astype(np.float32)
        norm = np.linalg.norm(histogram)
        if norm > 0:
            histogram = histogram / norm
            
        return histogram


class VLADEncoder(BaseEncoder):
    """
    Vector of Locally Aggregated Descriptors (VLAD) encoding.
    
    Encodes local descriptors by computing residuals from cluster centers
    and aggregating them into a fixed-length representation.
    """
    
    def __init__(self,
                 n_clusters=256,
                 power_normalization=True,
                 l2_normalization=True,
                 random_state=42,
                 **kwargs):
        """
        Initialize VLAD encoder.
        
        Parameters:
        -----------
        n_clusters : int
            Number of cluster centers
        power_normalization : bool
            Apply power normalization (signed square root)
        l2_normalization : bool
            Apply L2 normalization
        random_state : int
            Random seed
        """
        super().__init__(
            n_clusters=n_clusters,
            power_normalization=power_normalization,
            l2_normalization=l2_normalization,
            random_state=random_state,
            **kwargs
        )
        self.kmeans = None
        self.scaler = StandardScaler()
        
    def fit(self, features: np.ndarray) -> 'VLADEncoder':
        """Fit VLAD encoder by creating cluster centers."""
        if len(features) == 0:
            raise ValueError("No features provided for training!")
            
        print(f"Training VLAD with {len(features)} features...")
        
        # Normalize features
        features = self.scaler.fit_transform(features)
        
        # Create cluster centers
        print(f"Creating {self.params['n_clusters']} cluster centers...")
        self.kmeans = KMeans(
            n_clusters=self.params['n_clusters'],
            random_state=self.params['random_state'],
            n_init=10,
            max_iter=300
        )
        self.kmeans.fit(features)
        
        # VLAD dimension = n_clusters * descriptor_dimension
        descriptor_dim = features.shape[1]
        self._feature_dim = self.params['n_clusters'] * descriptor_dim
        self.is_trained = True
        print("VLAD training completed!")
        return self
        
    def encode(self, features: np.ndarray) -> np.ndarray:
        """Encode local features to VLAD representation."""
        if not self.is_trained:
            raise ValueError("VLAD not trained. Call fit() first.")
            
        if len(features) == 0:
            return np.zeros(self._feature_dim)
            
        # Normalize features
        features = self.scaler.transform(features)
        
        # Get cluster assignments and centers
        cluster_assignments = self.kmeans.predict(features)
        centers = self.kmeans.cluster_centers_
        
        # Compute VLAD representation
        vlad = np.zeros((self.params['n_clusters'], features.shape[1]))
        
        for i in range(self.params['n_clusters']):
            # Find descriptors assigned to cluster i
            cluster_mask = cluster_assignments == i
            
            if np.any(cluster_mask):
                # Compute residuals
                residuals = features[cluster_mask] - centers[i]
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


class FisherVectorEncoder(BaseEncoder):
    """
    Fisher Vector encoding.
    
    Encodes local descriptors using Fisher kernel with Gaussian Mixture Model.
    Simplified implementation focusing on first and second order statistics.
    """
    
    def __init__(self,
                 n_components=128,
                 power_normalization=True,
                 l2_normalization=True,
                 random_state=42,
                 **kwargs):
        """
        Initialize Fisher Vector encoder.
        
        Parameters:
        -----------
        n_components : int
            Number of Gaussian components
        power_normalization : bool
            Apply power normalization
        l2_normalization : bool
            Apply L2 normalization
        random_state : int
            Random seed
        """
        super().__init__(
            n_components=n_components,
            power_normalization=power_normalization,
            l2_normalization=l2_normalization,
            random_state=random_state,
            **kwargs
        )
        self.gmm = None
        self.scaler = StandardScaler()
        
    def fit(self, features: np.ndarray) -> 'FisherVectorEncoder':
        """Fit Fisher Vector encoder by training GMM."""
        if len(features) == 0:
            raise ValueError("No features provided for training!")
            
        print(f"Training Fisher Vector with {len(features)} features...")
        
        # Normalize features
        features = self.scaler.fit_transform(features)
        
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
            self.gmm.fit(features)
        
        # Fisher Vector dimension = 2 * n_components * descriptor_dimension
        descriptor_dim = features.shape[1]
        self._feature_dim = 2 * self.params['n_components'] * descriptor_dim
        self.is_trained = True
        print("Fisher Vector training completed!")
        return self
        
    def encode(self, features: np.ndarray) -> np.ndarray:
        """Encode local features to Fisher Vector."""
        if not self.is_trained:
            raise ValueError("Fisher Vector not trained. Call fit() first.")
            
        if len(features) == 0:
            return np.zeros(self._feature_dim)
            
        # Normalize features
        features = self.scaler.transform(features)
        
        # Compute Fisher Vector
        fv = self._compute_fisher_vector(features)
        
        # Apply normalizations
        if self.params['power_normalization']:
            fv = np.sign(fv) * np.sqrt(np.abs(fv))
            
        if self.params['l2_normalization']:
            norm = np.linalg.norm(fv)
            if norm > 0:
                fv = fv / norm
                
        return fv
        
    def _compute_fisher_vector(self, features: np.ndarray) -> np.ndarray:
        """Compute Fisher Vector for given features."""
        # Get GMM parameters
        weights = self.gmm.weights_
        means = self.gmm.means_
        covariances = self.gmm.covariances_
        
        # Compute posterior probabilities
        posteriors = self.gmm.predict_proba(features)  # Shape: (n_desc, n_components)
        
        # Initialize Fisher Vector
        n_components = self.params['n_components']
        descriptor_dim = features.shape[1]
        fv = np.zeros(2 * n_components * descriptor_dim)
        
        # Compute first and second order statistics
        for k in range(n_components):
            # First order statistic (mean deviation)
            diff = features - means[k]
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

import numpy as np
import pickle
from typing import Union, List, Optional, Tuple
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import warnings


class BaseEncoder:
    """Base class for descriptor encoders."""
    
    def __init__(self, **kwargs):
        """Initialize encoder with parameters."""
        self.params = kwargs
        self.is_trained = False
        self._feature_dim = None
    
    def fit(self, features: np.ndarray) -> 'BaseEncoder':
        """Fit encoder on feature set."""
        raise NotImplementedError("Subclasses must implement fit()")
    
    def encode(self, features: np.ndarray) -> np.ndarray:
        """Encode features to fixed-length vector."""
        raise NotImplementedError("Subclasses must implement encode()")
    
    def get_feature_dim(self) -> int:
        """Get dimension of encoded features."""
        return self._feature_dim
    
    def save(self, filepath: str) -> None:
        """Save encoder to file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    def load(self, filepath: str) -> None:
        """Load encoder from file."""
        with open(filepath, 'rb') as f:
            encoder = pickle.load(f)
            self.__dict__.update(encoder.__dict__)


class BagOfWordsEncoder(BaseEncoder):
    """
    Bag of Visual Words (BoVW) encoding.
    
    Converts local descriptors to fixed-length histograms using
    k-means clustering to create a visual vocabulary.
    """
    
    def __init__(self, 
                 n_clusters=512,
                 random_state=42,
                 **kwargs):
        """
        Initialize BoVW encoder.
        
        Parameters:
        -----------
        n_clusters : int
            Size of visual vocabulary (number of clusters)
        random_state : int
            Random seed for k-means
        """
        super().__init__(
            n_clusters=n_clusters,
            random_state=random_state,
            **kwargs
        )
        self.kmeans = None
        self.scaler = StandardScaler()
        
    def fit(self, features: np.ndarray) -> 'BagOfWordsEncoder':
        """
        Fit BoVW encoder by creating visual vocabulary.
        
        Parameters:
        -----------
        features : np.ndarray
            Local features for building vocabulary, shape (n_features, feature_dim)
        """
        if len(features) == 0:
            raise ValueError("No features provided for training!")
            
        print(f"Training BoVW with {len(features)} features...")
        
        # Normalize features
        features = self.scaler.fit_transform(features)
        
        # Create visual vocabulary using k-means
        print(f"Creating vocabulary with {self.params['n_clusters']} words...")
        self.kmeans = KMeans(
            n_clusters=self.params['n_clusters'],
            random_state=self.params['random_state'],
            n_init=10,
            max_iter=300
        )
        self.kmeans.fit(features)
        
        self._feature_dim = self.params['n_clusters']
        self.is_trained = True
        print("BoVW training completed!")
        return self
        
    def encode(self, features: np.ndarray) -> np.ndarray:
        """
        Encode local features to BoVW histogram.
        
        Parameters:
        -----------
        features : np.ndarray
            Local features to encode, shape (n_features, feature_dim)
            
        Returns:
        --------
        histogram : np.ndarray
            BoVW histogram of length n_clusters
        """
        if not self.is_trained:
            raise ValueError("BoVW not trained. Call fit() first.")
            
        if len(features) == 0:
            # No features, return zero histogram
            return np.zeros(self.params['n_clusters'])
            
        # Normalize features
        features = self.scaler.transform(features)
        
        # Assign features to clusters
        cluster_assignments = self.kmeans.predict(features)
        
        # Create histogram
        histogram = np.bincount(
            cluster_assignments, 
            minlength=self.params['n_clusters']
        )
        
        # Normalize histogram (L2 normalization)
        histogram = histogram.astype(np.float32)
        norm = np.linalg.norm(histogram)
        if norm > 0:
            histogram = histogram / norm
            
        return histogram