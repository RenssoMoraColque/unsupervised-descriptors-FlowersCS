"""
Classifiers and Cross-Validation
================================

Implementation of simple classifiers and cross-validation utilities
for descriptor evaluation.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings

from .metrics import ClassificationMetrics


class DescriptorClassifier:
    """
    Wrapper for training and evaluating classifiers on descriptor features.
    """
    
    def __init__(self, 
                 classifier_type: str = 'svm_linear',
                 classifier_params: Optional[Dict] = None,
                 normalize_features: bool = True,
                 random_state: int = 42):
        """
        Initialize descriptor classifier.
        
        Parameters:
        -----------
        classifier_type : str
            Type of classifier ('svm_linear', 'svm_rbf', 'knn', 'logistic')
        classifier_params : Dict, optional
            Parameters for the classifier
        normalize_features : bool
            Whether to normalize features with StandardScaler
        random_state : int
            Random seed
        """
        self.classifier_type = classifier_type
        self.classifier_params = classifier_params or {}
        self.normalize_features = normalize_features
        self.random_state = random_state
        
        # Initialize classifier and pipeline
        self.classifier = self._create_classifier()
        self.pipeline = self._create_pipeline()
        
    def _create_classifier(self):
        """Create classifier based on type."""
        if self.classifier_type == 'svm_linear':
            return SVC(
                kernel='linear',
                random_state=self.random_state,
                probability=True,  # Enable probability estimates
                **self.classifier_params
            )
        elif self.classifier_type == 'svm_rbf':
            return SVC(
                kernel='rbf',
                random_state=self.random_state,
                probability=True,
                **self.classifier_params
            )
        elif self.classifier_type == 'knn':
            return KNeighborsClassifier(**self.classifier_params)
        elif self.classifier_type == 'logistic':
            return LogisticRegression(
                random_state=self.random_state,
                **self.classifier_params
            )
        else:
            raise ValueError(f"Unknown classifier type: {self.classifier_type}")
    
    def _create_pipeline(self):
        """Create sklearn pipeline with optional normalization."""
        steps = []
        
        if self.normalize_features:
            steps.append(('scaler', StandardScaler()))
        
        steps.append(('classifier', self.classifier))
        
        return Pipeline(steps)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DescriptorClassifier':
        """
        Train the classifier.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix, shape (n_samples, n_features)
        y : np.ndarray
            Target labels
            
        Returns:
        --------
        self : DescriptorClassifier
            Fitted classifier
        """
        self.pipeline.fit(X, y)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
            
        Returns:
        --------
        predictions : np.ndarray
            Predicted labels
        """
        return self.pipeline.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
            
        Returns:
        --------
        probabilities : np.ndarray
            Class probabilities
        """
        if hasattr(self.pipeline, 'predict_proba'):
            return self.pipeline.predict_proba(X)
        else:
            # For classifiers without probability support
            predictions = self.predict(X)
            n_classes = len(np.unique(predictions))
            n_samples = len(predictions)
            probas = np.zeros((n_samples, n_classes))
            for i, pred in enumerate(predictions):
                probas[i, pred] = 1.0
            return probas
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute accuracy score.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            True labels
            
        Returns:
        --------
        accuracy : float
            Accuracy score
        """
        return self.pipeline.score(X, y)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, 
                class_names: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Comprehensive evaluation of the classifier.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            True labels
        class_names : List[str], optional
            Class names for detailed reporting
            
        Returns:
        --------
        metrics : Dict[str, float]
            Comprehensive evaluation metrics
        """
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        
        return ClassificationMetrics.compute_metrics(
            y, y_pred, y_proba, class_names
        )


class CrossValidationEvaluator:
    """
    Cross-validation evaluation for descriptors.
    """
    
    def __init__(self, 
                 n_splits: int = 5,
                 n_repeats: int = 3,
                 random_state: int = 42):
        """
        Initialize cross-validation evaluator.
        
        Parameters:
        -----------
        n_splits : int
            Number of CV folds
        n_repeats : int
            Number of repetitions with different random seeds
        random_state : int
            Base random seed
        """
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state
    
    def evaluate_descriptor(self,
                          X: np.ndarray,
                          y: np.ndarray,
                          classifier_configs: List[Dict],
                          class_names: Optional[List[str]] = None) -> Dict[str, Dict[str, Union[float, List[float]]]]:
        """
        Evaluate descriptor using cross-validation with multiple classifiers.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target labels
        classifier_configs : List[Dict]
            List of classifier configurations
        class_names : List[str], optional
            Class names
            
        Returns:
        --------
        results : Dict[str, Dict[str, Union[float, List[float]]]]
            CV results for each classifier configuration
        """
        results = {}
        
        for config in classifier_configs:
            config_name = self._get_config_name(config)
            print(f"Evaluating {config_name}...")
            
            # Collect results across repetitions
            all_scores = []
            all_detailed_metrics = []
            
            for repeat in range(self.n_repeats):
                # Use different random seed for each repetition
                seed = self.random_state + repeat
                
                # Create stratified k-fold
                skf = StratifiedKFold(
                    n_splits=self.n_splits,
                    shuffle=True,
                    random_state=seed
                )
                
                # Collect fold results
                fold_scores = []
                fold_detailed_metrics = []
                
                for train_idx, val_idx in skf.split(X, y):
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]
                    
                    # Create and train classifier
                    classifier = DescriptorClassifier(
                        classifier_type=config['type'],
                        classifier_params=config.get('params', {}),
                        random_state=seed
                    )
                    classifier.fit(X_train, y_train)
                    
                    # Evaluate on validation set
                    accuracy = classifier.score(X_val, y_val)
                    fold_scores.append(accuracy)
                    
                    # Detailed metrics
                    detailed_metrics = classifier.evaluate(X_val, y_val, class_names)
                    fold_detailed_metrics.append(detailed_metrics)
                
                all_scores.extend(fold_scores)
                all_detailed_metrics.extend(fold_detailed_metrics)
            
            # Aggregate results
            mean_accuracy = np.mean(all_scores)
            std_accuracy = np.std(all_scores)
            
            # Aggregate detailed metrics
            aggregated_metrics = self._aggregate_detailed_metrics(all_detailed_metrics)
            
            results[config_name] = {
                'accuracy_mean': mean_accuracy,
                'accuracy_std': std_accuracy,
                'accuracy_scores': all_scores,
                **aggregated_metrics
            }
            
            print(f"  {config_name}: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
        
        return results
    
    def hyperparameter_search(self,
                            X: np.ndarray,
                            y: np.ndarray,
                            classifier_type: str,
                            param_grid: Dict[str, List],
                            scoring: str = 'accuracy',
                            cv: int = 3) -> Tuple[Dict, float]:
        """
        Perform hyperparameter search using grid search CV.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target labels
        classifier_type : str
            Type of classifier
        param_grid : Dict[str, List]
            Parameter grid for search
        scoring : str
            Scoring metric
        cv : int
            Number of CV folds for grid search
            
        Returns:
        --------
        best_params : Dict
            Best hyperparameters
        best_score : float
            Best CV score
        """
        # Create base classifier
        base_classifier = DescriptorClassifier(
            classifier_type=classifier_type,
            random_state=self.random_state
        )
        
        # Prepare parameter grid for pipeline
        pipeline_param_grid = {}
        for param, values in param_grid.items():
            pipeline_param_grid[f'classifier__{param}'] = values
        
        # Grid search
        grid_search = GridSearchCV(
            base_classifier.pipeline,
            pipeline_param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, y)
        
        return grid_search.best_params_, grid_search.best_score_
    
    def _get_config_name(self, config: Dict) -> str:
        """Generate descriptive name for classifier configuration."""
        name = config['type']
        if 'params' in config and config['params']:
            param_strs = [f"{k}={v}" for k, v in config['params'].items()]
            name += f"({', '.join(param_strs)})"
        return name
    
    def _aggregate_detailed_metrics(self, 
                                  metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate detailed metrics across folds."""
        if not metrics_list:
            return {}
        
        # Get all metric names
        metric_names = set()
        for metrics in metrics_list:
            metric_names.update(metrics.keys())
        
        aggregated = {}
        for metric_name in metric_names:
            values = [m.get(metric_name, 0.0) for m in metrics_list]
            aggregated[f'{metric_name}_mean'] = np.mean(values)
            aggregated[f'{metric_name}_std'] = np.std(values)
        
        return aggregated