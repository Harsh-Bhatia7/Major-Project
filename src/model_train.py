import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier


class ModelTrainer:
    """Class to handle model training and evaluation"""

    def __init__(self, logger):
        """Initialize with logger"""
        self.logger = logger
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into training and test sets"""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        self.logger.info(f"Data split: {len(self.X_train)} training samples, {len(self.X_test)} testing samples")
        return self.X_train, self.X_test, self.y_train, self.y_test

    def train_decision_tree(self, X_train, y_train, max_depth=3, random_state=42):
        """Train a decision tree model"""
        self.logger.info("Step 5: Model Selection - Training Decision Tree")
        self.model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
        self.model.fit(X_train, y_train)
        self.logger.info("Model training completed")
        return self.model

    def perform_cross_validation(self, X, y, cv=5):
        """Perform cross-validation to check for potential overfitting"""
        self.logger.info("Performing cross-validation")
        dt_cv = DecisionTreeClassifier(random_state=42)
        cv_scores = cross_val_score(dt_cv, X, y, cv=cv)
        self.logger.info(f"Cross-validation scores: {cv_scores}")
        self.logger.info(f"Mean CV accuracy: {np.mean(cv_scores) * 100:.2f}%")
        self.logger.info(f"CV accuracy std dev: {np.std(cv_scores) * 100:.2f}%")
        return cv_scores
