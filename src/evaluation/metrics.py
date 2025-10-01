"""
Evaluation Metrics
==================

Implementation of various evaluation metrics for classification,
clustering, and retrieval tasks.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report,
    adjusted_rand_score, normalized_mutual_info_score,
    average_precision_score, roc_auc_score
)
from sklearn.metrics.cluster import adjusted_mutual_info_score
import warnings


class ClassificationMetrics:
    """
    Comprehensive classification metrics for descriptor evaluation.
    """
    
    @staticmethod
    def compute_metrics(y_true: np.ndarray, 
                       y_pred: np.ndarray,
                       y_proba: Optional[np.ndarray] = None,
                       class_names: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Compute comprehensive classification metrics.
        
        Parameters:
        -----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels
        y_proba : np.ndarray, optional
            Prediction probabilities (for mAP calculation)
        class_names : List[str], optional
            Class names for detailed reporting
            
        Returns:
        --------
        metrics : Dict[str, float]
            Dictionary containing all computed metrics
        """
        metrics = {}
        
        # Basic accuracy
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # Precision, Recall, F1-score
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        # Macro averages
        metrics['macro_precision'] = np.mean(precision)
        metrics['macro_recall'] = np.mean(recall)
        metrics['macro_f1'] = np.mean(f1)
        
        # Weighted averages
        precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        metrics['weighted_precision'] = precision_w
        metrics['weighted_recall'] = recall_w
        metrics['weighted_f1'] = f1_w
        
        # Per-class metrics
        if class_names is not None:
            for i, class_name in enumerate(class_names):
                if i < len(precision):
                    metrics[f'precision_{class_name}'] = precision[i]
                    metrics[f'recall_{class_name}'] = recall[i]
                    metrics[f'f1_{class_name}'] = f1[i]
                    metrics[f'support_{class_name}'] = support[i]
        
        # Mean Average Precision (if probabilities available)
        if y_proba is not None:
            try:
                n_classes = len(np.unique(y_true))
                if n_classes == 2:
                    # Binary classification
                    metrics['map'] = average_precision_score(y_true, y_proba[:, 1])
                    metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
                else:
                    # Multi-class: compute mAP using one-vs-rest
                    y_true_onehot = np.eye(n_classes)[y_true]
                    map_scores = []
                    for i in range(n_classes):
                        if np.sum(y_true_onehot[:, i]) > 0:  # Check if class exists
                            ap = average_precision_score(
                                y_true_onehot[:, i], y_proba[:, i]
                            )
                            map_scores.append(ap)
                    metrics['map'] = np.mean(map_scores) if map_scores else 0.0
                    
                    # Multi-class ROC AUC
                    try:
                        metrics['roc_auc'] = roc_auc_score(
                            y_true, y_proba, multi_class='ovr', average='macro'
                        )
                    except ValueError:
                        metrics['roc_auc'] = 0.0
                        
            except Exception as e:
                warnings.warn(f"Could not compute mAP/ROC-AUC: {e}")
                metrics['map'] = 0.0
                metrics['roc_auc'] = 0.0
        
        return metrics
    
    @staticmethod
    def print_classification_report(y_true: np.ndarray,
                                  y_pred: np.ndarray,
                                  class_names: Optional[List[str]] = None) -> str:
        """
        Generate detailed classification report.
        
        Parameters:
        -----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels
        class_names : List[str], optional
            Class names
            
        Returns:
        --------
        report : str
            Formatted classification report
        """
        return classification_report(
            y_true, y_pred, 
            target_names=class_names,
            zero_division=0
        )
    
    @staticmethod
    def compute_confusion_matrix(y_true: np.ndarray,
                               y_pred: np.ndarray,
                               normalize: Optional[str] = None) -> np.ndarray:
        """
        Compute confusion matrix.
        
        Parameters:
        -----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels
        normalize : str, optional
            Normalization mode ('true', 'pred', 'all', or None)
            
        Returns:
        --------
        cm : np.ndarray
            Confusion matrix
        """
        return confusion_matrix(y_true, y_pred, normalize=normalize)


class ClusteringMetrics:
    """
    Clustering evaluation metrics.
    """
    
    @staticmethod
    def compute_metrics(y_true: np.ndarray,
                       y_pred: np.ndarray) -> Dict[str, float]:
        """
        Compute clustering evaluation metrics.
        
        Parameters:
        -----------
        y_true : np.ndarray
            True cluster labels
        y_pred : np.ndarray
            Predicted cluster labels
            
        Returns:
        --------
        metrics : Dict[str, float]
            Dictionary containing clustering metrics
        """
        metrics = {}
        
        # Adjusted Rand Index
        metrics['ari'] = adjusted_rand_score(y_true, y_pred)
        
        # Normalized Mutual Information
        metrics['nmi'] = normalized_mutual_info_score(y_true, y_pred)
        
        # Adjusted Mutual Information
        metrics['ami'] = adjusted_mutual_info_score(y_true, y_pred)
        
        # Purity
        metrics['purity'] = ClusteringMetrics._compute_purity(y_true, y_pred)
        
        return metrics
    
    @staticmethod
    def _compute_purity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute purity score.
        
        Parameters:
        -----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted cluster labels
            
        Returns:
        --------
        purity : float
            Purity score
        """
        # Create contingency matrix
        contingency_matrix = np.zeros((len(np.unique(y_true)), len(np.unique(y_pred))))
        
        for i, true_label in enumerate(np.unique(y_true)):
            for j, pred_label in enumerate(np.unique(y_pred)):
                contingency_matrix[i, j] = np.sum(
                    (y_true == true_label) & (y_pred == pred_label)
                )
        
        # Compute purity
        purity = np.sum(np.max(contingency_matrix, axis=0)) / len(y_true)
        return purity


class RetrievalMetrics:
    """
    Information retrieval metrics for descriptor similarity.
    """
    
    @staticmethod
    def compute_metrics(similarity_matrix: np.ndarray,
                       y_true: np.ndarray,
                       k_values: List[int] = [1, 5, 10, 20]) -> Dict[str, float]:
        """
        Compute retrieval metrics.
        
        Parameters:
        -----------
        similarity_matrix : np.ndarray
            Pairwise similarity matrix between samples
        y_true : np.ndarray
            True labels for relevance determination
        k_values : List[int]
            Values of k for top-k metrics
            
        Returns:
        --------
        metrics : Dict[str, float]
            Dictionary containing retrieval metrics
        """
        metrics = {}
        n_samples = len(y_true)
        
        # For each query, compute precision@k and recall@k
        precisions_at_k = {k: [] for k in k_values}
        recalls_at_k = {k: [] for k in k_values}
        
        for query_idx in range(n_samples):
            # Get similarities for this query (exclude self)
            similarities = similarity_matrix[query_idx].copy()
            similarities[query_idx] = -np.inf  # Exclude self
            
            # Get sorted indices (most similar first)
            sorted_indices = np.argsort(similarities)[::-1]
            
            # Get relevant items (same class as query)
            query_label = y_true[query_idx]
            relevant_items = np.where(y_true == query_label)[0]
            relevant_items = relevant_items[relevant_items != query_idx]  # Exclude self
            n_relevant = len(relevant_items)
            
            if n_relevant == 0:
                continue  # Skip if no relevant items
            
            # Compute precision@k and recall@k for different k values
            for k in k_values:
                if k >= n_samples:
                    continue
                    
                top_k_items = sorted_indices[:k]
                n_relevant_retrieved = len(np.intersect1d(top_k_items, relevant_items))
                
                precision_k = n_relevant_retrieved / k
                recall_k = n_relevant_retrieved / n_relevant
                
                precisions_at_k[k].append(precision_k)
                recalls_at_k[k].append(recall_k)
        
        # Average metrics across all queries
        for k in k_values:
            if precisions_at_k[k]:  # Check if we have valid computations
                metrics[f'precision_at_{k}'] = np.mean(precisions_at_k[k])
                metrics[f'recall_at_{k}'] = np.mean(recalls_at_k[k])
            else:
                metrics[f'precision_at_{k}'] = 0.0
                metrics[f'recall_at_{k}'] = 0.0
        
        # Mean Average Precision (mAP)
        average_precisions = []
        for query_idx in range(n_samples):
            similarities = similarity_matrix[query_idx].copy()
            similarities[query_idx] = -np.inf
            sorted_indices = np.argsort(similarities)[::-1]
            
            query_label = y_true[query_idx]
            relevant_items = np.where(y_true == query_label)[0]
            relevant_items = relevant_items[relevant_items != query_idx]
            
            if len(relevant_items) == 0:
                continue
            
            # Compute average precision for this query
            precisions = []
            n_relevant_found = 0
            
            for rank, item_idx in enumerate(sorted_indices, 1):
                if item_idx in relevant_items:
                    n_relevant_found += 1
                    precision = n_relevant_found / rank
                    precisions.append(precision)
            
            if precisions:
                average_precision = np.mean(precisions)
                average_precisions.append(average_precision)
        
        metrics['map'] = np.mean(average_precisions) if average_precisions else 0.0
        
        return metrics