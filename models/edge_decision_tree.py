"""Edge-Level Lightweight Decision Tree Intrusion Detector.

Phase 4 — Edge-Level Lightweight Intrusion Detection Using Decision Tree.

This module provides the EdgeDecisionTreeClassifier class, wrapping scikit-learn's
DecisionTreeClassifier for low-latency binary intrusion detection (Normal vs. Attack)
on Industry 5.0 edge devices.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


class EdgeDecisionTreeClassifier:
    """Lightweight Decision Tree classifier for Edge-level binary intrusion detection."""

    def __init__(
        self,
        criterion: str = "gini",
        splitter: str = "best",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[Union[int, float, str]] = None,
        random_state: int = 42,
        class_weight: Optional[Union[Dict, str]] = None,
    ):
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.class_weight = class_weight

        self.model = DecisionTreeClassifier(
            criterion=self.criterion,
            splitter=self.splitter,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
            class_weight=self.class_weight,
        )

        self.feature_names_: List[str] = []
        self.classes_: np.ndarray = np.array([0, 1])
        self.is_fitted: bool = False

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        feature_names: Optional[List[str]] = None,
    ) -> "EdgeDecisionTreeClassifier":
        """Fits the Decision Tree classifier strictly on training data."""
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X_arr = X.to_numpy()
        else:
            X_arr = np.asarray(X)
            if feature_names is not None:
                self.feature_names_ = list(feature_names)
            elif not self.feature_names_:
                self.feature_names_ = [f"feature_{i}" for i in range(X_arr.shape[1])]

        y_arr = np.asarray(y)

        self.model.fit(X_arr, y_arr)
        self.classes_ = self.model.classes_
        self.is_fitted = True
        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predicts binary class labels (0 = Normal, 1 = Attack)."""
        if not self.is_fitted:
            raise RuntimeError("EdgeDecisionTreeClassifier is not fitted yet.")
        if isinstance(X, pd.DataFrame):
            X_arr = X[self.feature_names_].to_numpy() if self.feature_names_ else X.to_numpy()
        else:
            X_arr = np.asarray(X)
        return self.model.predict(X_arr)

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predicts class probabilities [P(Normal), P(Attack)]."""
        if not self.is_fitted:
            raise RuntimeError("EdgeDecisionTreeClassifier is not fitted yet.")
        if isinstance(X, pd.DataFrame):
            X_arr = X[self.feature_names_].to_numpy() if self.feature_names_ else X.to_numpy()
        else:
            X_arr = np.asarray(X)
        return self.model.predict_proba(X_arr)

    @property
    def depth(self) -> int:
        """Returns the maximum depth of the fitted tree."""
        if not self.is_fitted:
            return 0
        return self.model.get_depth()

    @property
    def node_count(self) -> int:
        """Returns the total number of nodes in the fitted tree."""
        if not self.is_fitted:
            return 0
        return self.model.tree_.node_count

    @property
    def leaf_count(self) -> int:
        """Returns the number of terminal leaves in the fitted tree."""
        if not self.is_fitted:
            return 0
        return self.model.get_n_leaves()

    @property
    def feature_importances_(self) -> np.ndarray:
        """Returns Gini feature importances."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        return self.model.feature_importances_

    def save(self, filepath: Union[str, Path]) -> Path:
        """Serializes the EdgeDecisionTreeClassifier instance to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        return filepath

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "EdgeDecisionTreeClassifier":
        """Loads a serialized EdgeDecisionTreeClassifier instance."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        return joblib.load(filepath)
