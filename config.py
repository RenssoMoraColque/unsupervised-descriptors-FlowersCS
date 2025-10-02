# Configuration for the unsupervised descriptors project

# Data paths
DATA_ROOT = "data"
RAW_DATA_PATH = "data/raw"
PROCESSED_DATA_PATH = "data/processed"

# STL-10 dataset configuration
STL10_ROOT = "data/raw/stl10_binary"
STL10_SPLITS = {
    "unlabeled": "unlabeled",  # 100k images for unsupervised learning
    "train": "train",          # 5k labeled images for evaluation
    "test": "test"             # 8k labeled images for testing
}

# Image preprocessing
IMAGE_SIZE = (96, 96)  # STL-10 default size
CHANNELS = 3

# Descriptor dimensions constraint
MAX_DESCRIPTOR_DIM = 4096

# Evaluation configuration
N_SPLITS = 3  # Number of repetitions for experiments
TEST_SIZE = 0.2
RANDOM_SEEDS = [42, 123, 456]  # Fixed seeds for reproducibility

# Robustness evaluation transformations
ROBUSTNESS_TRANSFORMS = {
    "gaussian_blur": {"sigma": 1.5},
    "rotation": {"angle_range": (-15, 15)},
    "scale": {"scale_range": (0.8, 1.2)},
    "brightness": {"factor_range": (0.7, 1.3)},
    "contrast": {"factor_range": (0.7, 1.3)},
    "jpeg_compression": {"quality": 40}
}

# Clustering algorithms parameters
CLUSTERING_PARAMS = {
    "kmeans": {
        "n_clusters_range": [50, 100, 256, 512, 1024],
        "random_state": 42,
        "max_iter": 300
    },
    "gmm": {
        "n_components_range": [50, 100, 256, 512],
        "random_state": 42,
        "max_iter": 100
    }
}

# Descriptor-specific parameters
DESCRIPTOR_PARAMS = {
    "hog": {
        "pixels_per_cell": (8, 8),
        "cells_per_block": (2, 2),
        "orientations": 9,
        "feature_vector": True
    },
    "sift": {
        "nfeatures": 0,  # No limit
        "nOctaveLayers": 3,
        "contrastThreshold": 0.04,
        "edgeThreshold": 10,
        "sigma": 1.6
    },
    "surf": {
        "hessianThreshold": 400,
        "nOctaves": 4,
        "nOctaveLayers": 3
    },
    "orb": {
        "nfeatures": 500,
        "scaleFactor": 1.2,
        "nlevels": 8
    },
    "lbp": {
        "radius": 3,
        "n_points": 24,
        "method": "uniform"
    }
}

# Bag of Visual Words parameters
BOVW_PARAMS = {
    "codebook_size": [128, 256, 512, 1024],
    "max_iterations": 100,
    "descriptor_per_image": 100  # Max descriptors to sample per image
}

# VLAD parameters
VLAD_PARAMS = {
    "codebook_size": [64, 128, 256],
    "power_normalization": True,
    "l2_normalization": True
}

# Dimensionality reduction
DIMRED_PARAMS = {
    "pca": {
        "n_components": [256, 512, 1024],
        "whiten": True
    },
    "ica": {
        "n_components": [256, 512],
        "max_iter": 200
    }
}

# Classifier parameters for evaluation
CLASSIFIER_PARAMS = {
    "svm_linear": {
        "C": [0.1, 1.0, 10.0],
        "max_iter": 1000
    },
    "svm_rbf": {
        "C": [0.1, 1.0, 10.0],
        "gamma": ["scale", "auto", 0.001, 0.01, 0.1]
    },
    "knn": {
        "n_neighbors": [3, 5, 7, 11],
        "weights": ["uniform", "distance"]
    },
    "logistic": {
        "C": [0.1, 1.0, 10.0],
        "max_iter": 1000
    }
}

# Results and logging
RESULTS_DIR = "results"
MODELS_DIR = "results/models"
METRICS_DIR = "results/metrics"
LOG_LEVEL = "INFO"