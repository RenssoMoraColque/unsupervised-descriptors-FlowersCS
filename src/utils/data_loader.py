"""
Data Loading Utilities
======================

Utilities for loading and splitting the STL-10 dataset.
"""

import os
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
from PIL import Image
import pickle
from torchvision.datasets import STL10
from torchvision import transforms
from sklearn.model_selection import train_test_split
import warnings


class STL10DataLoader:
    """
    Loader for STL-10 dataset with support for different splits.
    """
    
    def __init__(self, 
                 data_root: str = "data/raw",
                 download: bool = True,
                 transform: Optional[transforms.Compose] = None):
        """
        Initialize STL-10 data loader.
        
        Parameters:
        -----------
        data_root : str
            Root directory for data storage
        download : bool
            Whether to download dataset if not present
        transform : transforms.Compose, optional
            Transformations to apply to images
        """
        self.data_root = data_root
        self.download = download
        self.transform = transform or transforms.Compose([
            transforms.ToTensor()
        ])
        
        # Create data directory
        os.makedirs(data_root, exist_ok=True)
        
    def load_unlabeled(self) -> List[Union[np.ndarray, Image.Image]]:
        """
        Load unlabeled split (100k images) for unsupervised learning.
        
        Returns:
        --------
        images : List
            List of unlabeled images
        """
        print("Loading STL-10 unlabeled split...")
        dataset = STL10(
            root=self.data_root,
            split='unlabeled',
            download=self.download,
            transform=self.transform
        )
        
        images = []
        for i in range(len(dataset)):
            if i % 10000 == 0:
                print(f"Loaded {i}/{len(dataset)} unlabeled images")
            image, _ = dataset[i]  # No labels for unlabeled split
            images.append(image)
        
        print(f"Loaded {len(images)} unlabeled images")
        return images
    
    def load_labeled_split(self, split: str = 'train') -> Tuple[List, np.ndarray]:
        """
        Load labeled split (train or test).
        
        Parameters:
        -----------
        split : str
            Split to load ('train' or 'test')
            
        Returns:
        --------
        images : List
            List of images
        labels : np.ndarray
            Corresponding labels
        """
        print(f"Loading STL-10 {split} split...")
        dataset = STL10(
            root=self.data_root,
            split=split,
            download=self.download,
            transform=self.transform
        )
        
        images = []
        labels = []
        
        for i in range(len(dataset)):
            if i % 1000 == 0:
                print(f"Loaded {i}/{len(dataset)} {split} images")
            image, label = dataset[i]
            # Convert PIL image to numpy array
            if hasattr(image, 'numpy'):
                # If it's a tensor, convert to numpy
                image = image.numpy()
                # Convert from CHW to HWC format
                if image.ndim == 3 and image.shape[0] == 3:
                    image = np.transpose(image, (1, 2, 0))
            elif isinstance(image, Image.Image):
                # If it's still a PIL image, convert to numpy
                image = np.array(image)
            images.append(image)
            labels.append(label)
        
        labels = np.array(labels)
        print(f"Loaded {len(images)} {split} images with {len(np.unique(labels))} classes")
        return images, labels
    
    def load_train_test(self) -> Tuple[List, np.ndarray, List, np.ndarray]:
        """
        Load both train and test splits.
        
        Returns:
        --------
        train_images : List
            Training images
        train_labels : np.ndarray
            Training labels
        test_images : List
            Test images
        test_labels : np.ndarray
            Test labels
        """
        train_images, train_labels = self.load_labeled_split('train')
        test_images, test_labels = self.load_labeled_split('test')
        
        return train_images, train_labels, test_images, test_labels
    
    def get_class_names(self) -> List[str]:
        """
        Get STL-10 class names.
        
        Returns:
        --------
        class_names : List[str]
            List of class names
        """
        # STL-10 class names
        return [
            'airplane', 'bird', 'car', 'cat', 'deer',
            'dog', 'horse', 'monkey', 'ship', 'truck'
        ]
    
    def get_dataset_info(self) -> Dict[str, Union[int, List[str]]]:
        """
        Get information about the STL-10 dataset.
        
        Returns:
        --------
        info : Dict
            Dataset information
        """
        return {
            'n_classes': 10,
            'n_unlabeled': 100000,
            'n_train': 5000,
            'n_test': 8000,
            'image_size': (96, 96),
            'channels': 3,
            'class_names': self.get_class_names()
        }


class DatasetSplitter:
    """
    Utility for creating custom dataset splits.
    """
    
    @staticmethod
    def create_validation_split(images: List,
                              labels: np.ndarray,
                              val_size: float = 0.2,
                              random_state: int = 42,
                              stratify: bool = True) -> Tuple[List, List, np.ndarray, np.ndarray]:
        """
        Create train/validation split from labeled data.
        
        Parameters:
        -----------
        images : List
            Input images
        labels : np.ndarray
            Corresponding labels
        val_size : float
            Fraction of data for validation
        random_state : int
            Random seed for reproducibility
        stratify : bool
            Whether to maintain class distribution
            
        Returns:
        --------
        train_images : List
            Training images
        val_images : List
            Validation images
        train_labels : np.ndarray
            Training labels
        val_labels : np.ndarray
            Validation labels
        """
        indices = np.arange(len(images))
        stratify_labels = labels if stratify else None
        
        train_idx, val_idx = train_test_split(
            indices,
            test_size=val_size,
            random_state=random_state,
            stratify=stratify_labels
        )
        
        train_images = [images[i] for i in train_idx]
        val_images = [images[i] for i in val_idx]
        train_labels = labels[train_idx]
        val_labels = labels[val_idx]
        
        return train_images, val_images, train_labels, val_labels
    
    @staticmethod
    def create_subset(images: List,
                     labels: np.ndarray,
                     n_samples_per_class: int,
                     random_state: int = 42) -> Tuple[List, np.ndarray]:
        """
        Create balanced subset with specified samples per class.
        
        Parameters:
        -----------
        images : List
            Input images
        labels : np.ndarray
            Corresponding labels
        n_samples_per_class : int
            Number of samples per class
        random_state : int
            Random seed
            
        Returns:
        --------
        subset_images : List
            Subset images
        subset_labels : np.ndarray
            Subset labels
        """
        np.random.seed(random_state)
        
        unique_classes = np.unique(labels)
        subset_indices = []
        
        for class_label in unique_classes:
            class_indices = np.where(labels == class_label)[0]
            
            if len(class_indices) < n_samples_per_class:
                warnings.warn(f"Class {class_label} has only {len(class_indices)} samples, "
                            f"requested {n_samples_per_class}")
                selected_indices = class_indices
            else:
                selected_indices = np.random.choice(
                    class_indices, 
                    n_samples_per_class,
                    replace=False
                )
            
            subset_indices.extend(selected_indices)
        
        subset_indices = np.array(subset_indices)
        np.random.shuffle(subset_indices)  # Shuffle to mix classes
        
        subset_images = [images[i] for i in subset_indices]
        subset_labels = labels[subset_indices]
        
        return subset_images, subset_labels
    
    @staticmethod
    def combine_splits(*data_splits) -> Tuple[List, np.ndarray]:
        """
        Combine multiple data splits into one.
        
        Parameters:
        -----------
        data_splits : Tuple[List, np.ndarray]
            Multiple (images, labels) tuples to combine
            
        Returns:
        --------
        combined_images : List
            Combined images
        combined_labels : np.ndarray
            Combined labels
        """
        all_images = []
        all_labels = []
        
        for images, labels in data_splits:
            all_images.extend(images)
            all_labels.extend(labels)
        
        return all_images, np.array(all_labels)
    
    @staticmethod
    def save_split(images: List,
                  labels: np.ndarray,
                  filepath: str) -> None:
        """
        Save data split to disk.
        
        Parameters:
        -----------
        images : List
            Images to save
        labels : np.ndarray
            Labels to save
        filepath : str
            Path to save file
        """
        data = {'images': images, 'labels': labels}
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Saved split with {len(images)} samples to {filepath}")
    
    @staticmethod
    def load_split(filepath: str) -> Tuple[List, np.ndarray]:
        """
        Load data split from disk.
        
        Parameters:
        -----------
        filepath : str
            Path to saved file
            
        Returns:
        --------
        images : List
            Loaded images
        labels : np.ndarray
            Loaded labels
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        images = data['images']
        labels = data['labels']
        
        print(f"Loaded split with {len(images)} samples from {filepath}")
        return images, labels