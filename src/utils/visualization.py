"""
Visualization Utilities
======================

Utilities for visualizing results, analyzing descriptors, and creating plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Union, Optional, Tuple, Any
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import warnings

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class ResultsVisualizer:
    """
    Visualization utilities for evaluation results.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize results visualizer.
        
        Parameters:
        -----------
        figsize : tuple
            Default figure size for plots
        """
        self.figsize = figsize
        
    def plot_performance_comparison(self, 
                                  results: Dict[str, Dict],
                                  metric: str = 'accuracy',
                                  save_path: Optional[str] = None) -> None:
        """
        Plot performance comparison across descriptors.
        
        Parameters:
        -----------
        results : Dict
            Results dictionary from evaluation
        metric : str
            Metric to plot ('accuracy', 'macro_f1', etc.)
        save_path : str, optional
            Path to save the plot
        """
        # Extract data for plotting
        descriptors = []
        scores = []
        
        for desc_name, desc_results in results.items():
            if 'classifiers' in desc_results:
                # Find best classifier for this descriptor
                best_score = 0
                for clf_results in desc_results['classifiers'].values():
                    if 'test_metrics' in clf_results:
                        score = clf_results['test_metrics'].get(metric, 0)
                        if score > best_score:
                            best_score = score
                
                descriptors.append(desc_name.replace('_', ' ').title())
                scores.append(best_score)
        
        # Create plot
        plt.figure(figsize=self.figsize)
        bars = plt.bar(descriptors, scores, alpha=0.8)
        
        # Customize plot
        plt.title(f'Descriptor Performance Comparison ({metric.replace("_", " ").title()})', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Descriptors', fontsize=12)
        plt.ylabel(metric.replace('_', ' ').title(), fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()
    
    def plot_robustness_analysis(self, 
                                robustness_results: Dict[str, Dict],
                                save_path: Optional[str] = None) -> None:
        """
        Plot robustness analysis results.
        
        Parameters:
        -----------
        robustness_results : Dict
            Robustness results from evaluation
        save_path : str, optional
            Path to save the plot
        """
        if not robustness_results:
            print("No robustness results to plot.")
            return
        
        # Prepare data
        descriptors = list(robustness_results.keys())
        transform_names = set()
        
        for results in robustness_results.values():
            transform_names.update(results.keys())
        transform_names.discard('baseline')
        transform_names = sorted(transform_names)
        
        # Create data matrix
        data_matrix = []
        for desc_name in descriptors:
            row = []
            for transform in transform_names:
                if transform in robustness_results[desc_name]:
                    drop = robustness_results[desc_name][transform].get('drop', 0)
                    row.append(abs(drop))  # Use absolute value for better visualization
                else:
                    row.append(0)
            data_matrix.append(row)
        
        # Create heatmap
        plt.figure(figsize=(max(len(transform_names) * 1.5, 10), max(len(descriptors) * 0.8, 6)))
        
        df = pd.DataFrame(data_matrix, 
                         index=[desc.replace('_', ' ').title() for desc in descriptors],
                         columns=[t.replace('_', ' ').title() for t in transform_names])
        
        sns.heatmap(df, annot=True, fmt='.3f', cmap='Reds', 
                   cbar_kws={'label': 'Accuracy Drop'})
        
        plt.title('Robustness Analysis: Accuracy Drop per Transformation', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Transformations', fontsize=12)
        plt.ylabel('Descriptors', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Robustness plot saved to: {save_path}")
        
        plt.show()
    
    def plot_efficiency_analysis(self, 
                               results: Dict[str, Dict],
                               save_path: Optional[str] = None) -> None:
        """
        Plot efficiency analysis (time vs accuracy).
        
        Parameters:
        -----------
        results : Dict
            Results dictionary from evaluation
        save_path : str, optional
            Path to save the plot
        """
        # Extract data
        descriptors = []
        accuracies = []
        times = []
        dimensions = []
        
        for desc_name, desc_results in results.items():
            if 'feature_info' in desc_results and 'classifiers' in desc_results:
                # Get best accuracy
                best_accuracy = 0
                for clf_results in desc_results['classifiers'].values():
                    if 'test_metrics' in clf_results:
                        acc = clf_results['test_metrics'].get('accuracy', 0)
                        if acc > best_accuracy:
                            best_accuracy = acc
                
                # Get timing info
                time_per_img = desc_results['feature_info'].get('avg_time_per_image', 0) * 1000  # Convert to ms
                dims = desc_results['feature_info'].get('dimensions', 0)
                
                descriptors.append(desc_name.replace('_', ' ').title())
                accuracies.append(best_accuracy)
                times.append(time_per_img)
                dimensions.append(dims)
        
        # Create scatter plot
        plt.figure(figsize=self.figsize)
        
        # Create scatter plot with bubble size based on dimensions
        scatter = plt.scatter(times, accuracies, s=[d/10 for d in dimensions], 
                            alpha=0.7, c=range(len(descriptors)), cmap='viridis')
        
        # Add labels for each point
        for i, desc in enumerate(descriptors):
            plt.annotate(desc, (times[i], accuracies[i]), 
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=10, alpha=0.8)
        
        plt.xlabel('Time per Image (ms)', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Efficiency Analysis: Accuracy vs Speed\n(Bubble size = Descriptor Dimensions)', 
                 fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Descriptor Index', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Efficiency plot saved to: {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, 
                            y_true: np.ndarray,
                            y_pred: np.ndarray,
                            class_names: List[str],
                            title: str = "Confusion Matrix",
                            save_path: Optional[str] = None) -> None:
        """
        Plot confusion matrix.
        
        Parameters:
        -----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels
        class_names : List[str]
            Class names
        title : str
            Plot title
        save_path : str, optional
            Path to save the plot
        """
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create plot
        plt.figure(figsize=(10, 8))
        
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', 
                   xticklabels=class_names, yticklabels=class_names,
                   cmap='Blues', cbar_kws={'label': 'Normalized Count'})
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('True', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to: {save_path}")
        
        plt.show()
    
    def plot_feature_importance(self,
                              feature_weights: np.ndarray,
                              feature_names: Optional[List[str]] = None,
                              top_k: int = 20,
                              title: str = "Feature Importance",
                              save_path: Optional[str] = None) -> None:
        """
        Plot feature importance/weights.
        
        Parameters:
        -----------
        feature_weights : np.ndarray
            Feature weights or importance scores
        feature_names : List[str], optional
            Names of features
        top_k : int
            Number of top features to show
        title : str
            Plot title
        save_path : str, optional
            Path to save the plot
        """
        # Get top k features
        abs_weights = np.abs(feature_weights)
        top_indices = np.argsort(abs_weights)[-top_k:][::-1]
        top_weights = feature_weights[top_indices]
        
        if feature_names is not None:
            top_names = [feature_names[i] for i in top_indices]
        else:
            top_names = [f"Feature {i}" for i in top_indices]
        
        # Create plot
        plt.figure(figsize=(max(top_k * 0.5, 8), 6))
        
        colors = ['red' if w < 0 else 'blue' for w in top_weights]
        bars = plt.barh(range(len(top_weights)), top_weights, color=colors, alpha=0.7)
        
        plt.yticks(range(len(top_weights)), top_names)
        plt.xlabel('Weight/Importance', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (bar, weight) in enumerate(zip(bars, top_weights)):
            plt.text(weight + 0.01 if weight >= 0 else weight - 0.01, 
                    bar.get_y() + bar.get_height()/2,
                    f'{weight:.3f}', ha='left' if weight >= 0 else 'right', 
                    va='center', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to: {save_path}")
        
        plt.show()


class DescriptorAnalyzer:
    """
    Analyzer for descriptor features and patterns.
    """
    
    def __init__(self):
        """Initialize descriptor analyzer."""
        pass
    
    def visualize_feature_distribution(self,
                                     features: np.ndarray,
                                     labels: Optional[np.ndarray] = None,
                                     class_names: Optional[List[str]] = None,
                                     method: str = 'tsne',
                                     save_path: Optional[str] = None) -> None:
        """
        Visualize feature distribution using dimensionality reduction.
        
        Parameters:
        -----------
        features : np.ndarray
            Feature matrix, shape (n_samples, n_features)
        labels : np.ndarray, optional
            Class labels for coloring
        class_names : List[str], optional
            Names of classes
        method : str
            Reduction method ('tsne', 'pca')
        save_path : str, optional
            Path to save the plot
        """
        # Reduce to 2D
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, features.shape[0]-1))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                features_2d = reducer.fit_transform(features)
        elif method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
            features_2d = reducer.fit_transform(features)
        else:
            raise ValueError(f"Unknown reduction method: {method}")
        
        # Create plot
        plt.figure(figsize=(10, 8))
        
        if labels is not None:
            # Color by class
            unique_labels = np.unique(labels)
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
            
            for i, label in enumerate(unique_labels):
                mask = labels == label
                label_name = class_names[label] if class_names else f"Class {label}"
                plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                          c=[colors[i]], label=label_name, alpha=0.7, s=50)
            
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            # No labels
            plt.scatter(features_2d[:, 0], features_2d[:, 1], alpha=0.7, s=50)
        
        plt.xlabel(f'{method.upper()} Component 1', fontsize=12)
        plt.ylabel(f'{method.upper()} Component 2', fontsize=12)
        plt.title(f'Feature Distribution Visualization ({method.upper()})', 
                 fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature distribution plot saved to: {save_path}")
        
        plt.show()
    
    def analyze_feature_statistics(self,
                                 features: np.ndarray,
                                 feature_names: Optional[List[str]] = None,
                                 save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze statistical properties of features.
        
        Parameters:
        -----------
        features : np.ndarray
            Feature matrix
        feature_names : List[str], optional
            Names of features
        save_path : str, optional
            Path to save analysis report
            
        Returns:
        --------
        stats : Dict
            Statistical analysis results
        """
        # Compute statistics
        stats = {
            'mean': np.mean(features, axis=0),
            'std': np.std(features, axis=0),
            'min': np.min(features, axis=0),
            'max': np.max(features, axis=0),
            'median': np.median(features, axis=0),
            'skewness': self._compute_skewness(features),
            'kurtosis': self._compute_kurtosis(features)
        }
        
        # Overall statistics
        overall_stats = {
            'n_samples': features.shape[0],
            'n_features': features.shape[1],
            'sparsity': np.mean(features == 0),
            'dynamic_range': np.mean(stats['max'] - stats['min']),
            'mean_correlation': np.mean(np.abs(np.corrcoef(features.T))) if features.shape[1] > 1 else 0
        }
        
        # Print summary
        print("Feature Statistics Summary:")
        print("=" * 40)
        print(f"Number of samples: {overall_stats['n_samples']}")
        print(f"Number of features: {overall_stats['n_features']}")
        print(f"Sparsity: {overall_stats['sparsity']:.3f}")
        print(f"Average dynamic range: {overall_stats['dynamic_range']:.3f}")
        print(f"Mean absolute correlation: {overall_stats['mean_correlation']:.3f}")
        
        print(f"\nFeature value ranges:")
        print(f"Mean ± Std: {np.mean(stats['mean']):.3f} ± {np.mean(stats['std']):.3f}")
        print(f"Global min/max: {np.min(stats['min']):.3f} / {np.max(stats['max']):.3f}")
        
        # Create distribution plot
        if features.shape[1] <= 50:  # Only plot if not too many features
            plt.figure(figsize=(12, 8))
            
            # Plot feature distributions
            plt.subplot(2, 2, 1)
            plt.hist(stats['mean'], bins=20, alpha=0.7, edgecolor='black')
            plt.title('Distribution of Feature Means')
            plt.xlabel('Mean Value')
            
            plt.subplot(2, 2, 2)
            plt.hist(stats['std'], bins=20, alpha=0.7, edgecolor='black')
            plt.title('Distribution of Feature Standard Deviations')
            plt.xlabel('Std Value')
            
            plt.subplot(2, 2, 3)
            plt.hist(stats['skewness'], bins=20, alpha=0.7, edgecolor='black')
            plt.title('Distribution of Feature Skewness')
            plt.xlabel('Skewness')
            
            plt.subplot(2, 2, 4)
            plt.hist(stats['kurtosis'], bins=20, alpha=0.7, edgecolor='black')
            plt.title('Distribution of Feature Kurtosis')
            plt.xlabel('Kurtosis')
            
            plt.tight_layout()
            
            if save_path:
                plot_path = save_path.replace('.txt', '_distributions.png')
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                print(f"Distribution plots saved to: {plot_path}")
            
            plt.show()
        
        # Save detailed analysis
        if save_path:
            with open(save_path, 'w') as f:
                f.write("Feature Statistics Analysis\n")
                f.write("=" * 40 + "\n\n")
                
                f.write("Overall Statistics:\n")
                for key, value in overall_stats.items():
                    f.write(f"{key}: {value}\n")
                
                f.write("\nPer-Feature Statistics:\n")
                if feature_names:
                    for i, name in enumerate(feature_names):
                        f.write(f"{name}: mean={stats['mean'][i]:.3f}, "
                               f"std={stats['std'][i]:.3f}, "
                               f"range=[{stats['min'][i]:.3f}, {stats['max'][i]:.3f}]\n")
            
            print(f"Detailed analysis saved to: {save_path}")
        
        return {**stats, 'overall': overall_stats}
    
    def _compute_skewness(self, features: np.ndarray) -> np.ndarray:
        """Compute skewness for each feature."""
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        normalized = (features - mean) / (std + 1e-8)
        return np.mean(normalized ** 3, axis=0)
    
    def _compute_kurtosis(self, features: np.ndarray) -> np.ndarray:
        """Compute kurtosis for each feature."""
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        normalized = (features - mean) / (std + 1e-8)
        return np.mean(normalized ** 4, axis=0) - 3  # Excess kurtosis