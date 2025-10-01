"""
Image Descriptors Module
========================

This module contains implementations of various classical image descriptors
including global descriptors (HOG, LBP, color histograms) and local 
descriptors with encoding schemes (SIFT+BoVW, VLAD, Fisher Vectors).
"""

from .global_descriptors import (
    HOGDescriptor,
    LBPDescriptor,
    ColorHistogramDescriptor,
    GISTDescriptor
)

from .local_descriptors import (
    SIFTDescriptor,
    SURFDescriptor,
    ORBDescriptor,
    BRISKDescriptor
)

from .encoding import (
    BagOfWordsEncoder,
    VLADEncoder,
    FisherVectorEncoder
)

from .base import BaseDescriptor

__all__ = [
    "BaseDescriptor",
    "HOGDescriptor",
    "LBPDescriptor", 
    "ColorHistogramDescriptor",
    "GISTDescriptor",
    "SIFTDescriptor",
    "SURFDescriptor",
    "ORBDescriptor",
    "BRISKDescriptor",
    "BagOfWordsEncoder",
    "VLADEncoder",
    "FisherVectorEncoder"
]