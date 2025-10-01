"""
Robustness Evaluation
====================

Implementation of robustness testing with various image transformations.
"""

import numpy as np
from typing import List, Dict, Tuple, Union, Callable, Optional
from PIL import Image, ImageFilter, ImageEnhance
import cv2
from io import BytesIO
import warnings

from ..descriptors.base import BaseDescriptor


class ImageTransforms:
    """
    Collection of image transformation functions for robustness testing.
    """
    
    @staticmethod
    def gaussian_blur(image: Union[np.ndarray, Image.Image], 
                     sigma: float = 1.5) -> Union[np.ndarray, Image.Image]:
        """
        Apply Gaussian blur to image.
        
        Parameters:
        -----------
        image : np.ndarray or PIL.Image
            Input image
        sigma : float
            Standard deviation for Gaussian kernel
            
        Returns:
        --------
        blurred_image : Same type as input
            Blurred image
        """
        if isinstance(image, Image.Image):
            # PIL Image
            return image.filter(ImageFilter.GaussianBlur(radius=sigma))
        else:
            # NumPy array
            if len(image.shape) == 3:
                # Color image
                blurred = cv2.GaussianBlur(image, (0, 0), sigma)
            else:
                # Grayscale
                blurred = cv2.GaussianBlur(image, (0, 0), sigma)
            return blurred
    
    @staticmethod
    def rotation(image: Union[np.ndarray, Image.Image],
                angle: float) -> Union[np.ndarray, Image.Image]:
        """
        Rotate image by given angle.
        
        Parameters:
        -----------
        image : np.ndarray or PIL.Image
            Input image
        angle : float
            Rotation angle in degrees
            
        Returns:
        --------
        rotated_image : Same type as input
            Rotated image
        """
        if isinstance(image, Image.Image):
            # PIL Image
            return image.rotate(angle, expand=True, fillcolor=(128, 128, 128))
        else:
            # NumPy array
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            
            # Get rotation matrix
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # Compute new dimensions
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])
            new_w = int((h * sin) + (w * cos))
            new_h = int((h * cos) + (w * sin))
            
            # Adjust translation
            M[0, 2] += (new_w / 2) - center[0]
            M[1, 2] += (new_h / 2) - center[1]
            
            # Apply rotation
            if len(image.shape) == 3:
                rotated = cv2.warpAffine(image, M, (new_w, new_h), 
                                       borderValue=(128, 128, 128))
            else:
                rotated = cv2.warpAffine(image, M, (new_w, new_h), 
                                       borderValue=128)
            return rotated
    
    @staticmethod
    def scale(image: Union[np.ndarray, Image.Image],
              scale_factor: float) -> Union[np.ndarray, Image.Image]:
        """
        Scale image by given factor.
        
        Parameters:
        -----------
        image : np.ndarray or PIL.Image
            Input image
        scale_factor : float
            Scaling factor (1.0 = no change)
            
        Returns:
        --------
        scaled_image : Same type as input
            Scaled image
        """
        if isinstance(image, Image.Image):
            # PIL Image
            w, h = image.size
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)
            return image.resize((new_w, new_h), Image.LANCZOS)
        else:
            # NumPy array
            h, w = image.shape[:2]
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)
            return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    @staticmethod
    def brightness(image: Union[np.ndarray, Image.Image],
                  factor: float) -> Union[np.ndarray, Image.Image]:
        """
        Adjust image brightness.
        
        Parameters:
        -----------
        image : np.ndarray or PIL.Image
            Input image
        factor : float
            Brightness factor (1.0 = no change, >1.0 = brighter, <1.0 = darker)
            
        Returns:
        --------
        adjusted_image : Same type as input
            Brightness-adjusted image
        """
        if isinstance(image, Image.Image):
            # PIL Image
            enhancer = ImageEnhance.Brightness(image)
            return enhancer.enhance(factor)
        else:
            # NumPy array
            if image.dtype == np.uint8:
                adjusted = np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)
            else:
                adjusted = np.clip(image * factor, 0, 1)
            return adjusted
    
    @staticmethod
    def contrast(image: Union[np.ndarray, Image.Image],
                factor: float) -> Union[np.ndarray, Image.Image]:
        """
        Adjust image contrast.
        
        Parameters:
        -----------
        image : np.ndarray or PIL.Image
            Input image
        factor : float
            Contrast factor (1.0 = no change, >1.0 = more contrast, <1.0 = less contrast)
            
        Returns:
        --------
        adjusted_image : Same type as input
            Contrast-adjusted image
        """
        if isinstance(image, Image.Image):
            # PIL Image
            enhancer = ImageEnhance.Contrast(image)
            return enhancer.enhance(factor)
        else:
            # NumPy array
            if image.dtype == np.uint8:
                mean_val = np.mean(image)
                adjusted = np.clip(
                    mean_val + factor * (image.astype(np.float32) - mean_val), 
                    0, 255
                ).astype(np.uint8)
            else:
                mean_val = np.mean(image)
                adjusted = np.clip(mean_val + factor * (image - mean_val), 0, 1)
            return adjusted
    
    @staticmethod
    def jpeg_compression(image: Union[np.ndarray, Image.Image],
                        quality: int = 40) -> Union[np.ndarray, Image.Image]:
        """
        Apply JPEG compression artifacts.
        
        Parameters:
        -----------
        image : np.ndarray or PIL.Image
            Input image
        quality : int
            JPEG quality (1-100, lower = more compression)
            
        Returns:
        --------
        compressed_image : Same type as input
            JPEG-compressed image
        """
        # Convert to PIL if necessary
        if isinstance(image, np.ndarray):
            if image.dtype == np.float32 or image.dtype == np.float64:
                image_pil = Image.fromarray((image * 255).astype(np.uint8))
            else:
                image_pil = Image.fromarray(image)
            return_numpy = True
        else:
            image_pil = image
            return_numpy = False
        
        # Apply JPEG compression
        buffer = BytesIO()
        image_pil.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        compressed_pil = Image.open(buffer)
        
        # Convert back if necessary
        if return_numpy:
            return np.array(compressed_pil)
        else:
            return compressed_pil


class RobustnessEvaluator:
    """
    Evaluator for testing descriptor robustness to various transformations.
    """
    
    def __init__(self, descriptor: BaseDescriptor):
        """
        Initialize robustness evaluator.
        
        Parameters:
        -----------
        descriptor : BaseDescriptor
            Trained descriptor to evaluate
        """
        self.descriptor = descriptor
        if not descriptor.is_trained:
            raise ValueError("Descriptor must be trained before robustness evaluation")
    
    def evaluate_robustness(self,
                          images: List[Union[np.ndarray, Image.Image]],
                          labels: np.ndarray,
                          classifier,
                          transform_params: Dict[str, Dict]) -> Dict[str, Dict[str, float]]:
        """
        Evaluate descriptor robustness to various transformations.
        
        Parameters:
        -----------
        images : List of images
            Test images
        labels : np.ndarray
            True labels for images
        classifier : sklearn classifier
            Trained classifier for evaluation
        transform_params : Dict[str, Dict]
            Parameters for each transformation type
            
        Returns:
        --------
        results : Dict[str, Dict[str, float]]
            Robustness results for each transformation
        """
        from .metrics import ClassificationMetrics
        
        # Extract baseline descriptors
        print("Extracting baseline descriptors...")
        baseline_descriptors = self.descriptor.extract_batch(images)
        baseline_pred = classifier.predict(baseline_descriptors)
        baseline_accuracy = ClassificationMetrics.compute_metrics(
            labels, baseline_pred
        )['accuracy']
        
        results = {
            'baseline': {'accuracy': baseline_accuracy, 'drop': 0.0}
        }
        
        # Test each transformation
        for transform_name, params in transform_params.items():
            print(f"Testing robustness to {transform_name}...")
            
            try:
                # Apply transformation
                transformed_images = self._apply_transformation(
                    images, transform_name, params
                )
                
                # Extract descriptors from transformed images
                transformed_descriptors = self.descriptor.extract_batch(transformed_images)
                
                # Classify transformed descriptors
                transformed_pred = classifier.predict(transformed_descriptors)
                
                # Compute metrics
                transformed_accuracy = ClassificationMetrics.compute_metrics(
                    labels, transformed_pred
                )['accuracy']
                
                # Compute accuracy drop
                accuracy_drop = baseline_accuracy - transformed_accuracy
                relative_drop = (accuracy_drop / baseline_accuracy) * 100 if baseline_accuracy > 0 else 0
                
                results[transform_name] = {
                    'accuracy': transformed_accuracy,
                    'drop': accuracy_drop,
                    'relative_drop_percent': relative_drop
                }
                
                print(f"  {transform_name}: {transformed_accuracy:.4f} "
                      f"(drop: {accuracy_drop:.4f}, {relative_drop:.1f}%)")
                
            except Exception as e:
                warnings.warn(f"Could not evaluate {transform_name}: {e}")
                results[transform_name] = {
                    'accuracy': 0.0,
                    'drop': baseline_accuracy,
                    'relative_drop_percent': 100.0,
                    'error': str(e)
                }
        
        return results
    
    def _apply_transformation(self,
                            images: List[Union[np.ndarray, Image.Image]],
                            transform_name: str,
                            params: Dict) -> List[Union[np.ndarray, Image.Image]]:
        """
        Apply specified transformation to all images.
        
        Parameters:
        -----------
        images : List of images
            Input images
        transform_name : str
            Name of transformation
        params : Dict
            Transformation parameters
            
        Returns:
        --------
        transformed_images : List of images
            Transformed images
        """
        transformed = []
        
        for img in images:
            if transform_name == 'gaussian_blur':
                transformed_img = ImageTransforms.gaussian_blur(
                    img, sigma=params.get('sigma', 1.5)
                )
            elif transform_name == 'rotation':
                # Random angle within range
                angle_range = params.get('angle_range', (-15, 15))
                angle = np.random.uniform(*angle_range)
                transformed_img = ImageTransforms.rotation(img, angle)
            elif transform_name == 'scale':
                # Random scale within range
                scale_range = params.get('scale_range', (0.8, 1.2))
                scale = np.random.uniform(*scale_range)
                transformed_img = ImageTransforms.scale(img, scale)
            elif transform_name == 'brightness':
                # Random brightness within range
                factor_range = params.get('factor_range', (0.7, 1.3))
                factor = np.random.uniform(*factor_range)
                transformed_img = ImageTransforms.brightness(img, factor)
            elif transform_name == 'contrast':
                # Random contrast within range
                factor_range = params.get('factor_range', (0.7, 1.3))
                factor = np.random.uniform(*factor_range)
                transformed_img = ImageTransforms.contrast(img, factor)
            elif transform_name == 'jpeg_compression':
                quality = params.get('quality', 40)
                transformed_img = ImageTransforms.jpeg_compression(img, quality)
            else:
                raise ValueError(f"Unknown transformation: {transform_name}")
            
            transformed.append(transformed_img)
        
        return transformed
    
    def compare_descriptors(self,
                          images: List[Union[np.ndarray, Image.Image]],
                          transform_params: Dict[str, Dict]) -> Dict[str, float]:
        """
        Compare descriptor similarity before and after transformations.
        
        Parameters:
        -----------
        images : List of images
            Test images
        transform_params : Dict[str, Dict]
            Transformation parameters
            
        Returns:
        --------
        similarities : Dict[str, float]
            Average cosine similarity for each transformation
        """
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Extract baseline descriptors
        baseline_descriptors = self.descriptor.extract_batch(images)
        
        similarities = {}
        
        for transform_name, params in transform_params.items():
            # Apply transformation
            transformed_images = self._apply_transformation(
                images, transform_name, params
            )
            
            # Extract transformed descriptors
            transformed_descriptors = self.descriptor.extract_batch(transformed_images)
            
            # Compute pairwise cosine similarities
            cos_similarities = []
            for i in range(len(images)):
                if i < len(baseline_descriptors) and i < len(transformed_descriptors):
                    sim = cosine_similarity(
                        baseline_descriptors[i:i+1],
                        transformed_descriptors[i:i+1]
                    )[0, 0]
                    cos_similarities.append(sim)
            
            # Average similarity
            avg_similarity = np.mean(cos_similarities) if cos_similarities else 0.0
            similarities[transform_name] = avg_similarity
        
        return similarities