import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


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
        self.available_models = {
            'decision_tree': self.get_decision_tree,
            'random_forest': self.get_random_forest,
            'svm': self.get_svm,
            'knn': self.get_knn
        }

    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into training and test sets"""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        self.logger.info(f"Data split: {len(self.X_train)} training samples, {len(self.X_test)} testing samples")
        return self.X_train, self.X_test, self.y_train, self.y_test

    def get_decision_tree(self, **kwargs):
        """Get a decision tree classifier with given parameters"""
        params = {
            'max_depth': 3,
            'random_state': 42
        }
        params.update(kwargs)
        return DecisionTreeClassifier(**params)

    def get_random_forest(self, **kwargs):
        """Get a random forest classifier with given parameters"""
        params = {
            'n_estimators': 100,
            'max_depth': 3,
            'random_state': 42
        }
        params.update(kwargs)
        return RandomForestClassifier(**params)

    def get_svm(self, **kwargs):
        """Get an SVM classifier with given parameters"""
        params = {
            'C': 1.0,
            'kernel': 'rbf',
            'random_state': 42
        }
        params.update(kwargs)
        return SVC(**params)

    def get_knn(self, **kwargs):
        """Get a KNN classifier with given parameters"""
        params = {
            'n_neighbors': 5,
            'weights': 'uniform'
        }
        params.update(kwargs)
        return KNeighborsClassifier(**params)

    def train_model(self, X_train, y_train, model_type='decision_tree', **model_params):
        """Train a model of the specified type with the given parameters"""
        if model_type not in self.available_models:
            self.logger.error(f"Unknown model type: {model_type}. Using decision_tree instead.")
            model_type = 'decision_tree'

        self.logger.info(f"Step 5: Model Selection - Training {model_type}")

        # Get the model instance with provided parameters
        model_func = self.available_models[model_type]
        self.model = model_func(**model_params)

        # Train the model
        self.model.fit(X_train, y_train)
        self.logger.info("Model training completed")
        return self.model

    def perform_cross_validation(self, X, y, model_type='decision_tree', cv=5, **model_params):
        """Perform cross-validation to check for potential overfitting"""
        self.logger.info(f"Performing cross-validation for {model_type}")

        # Get the model instance with provided parameters
        if model_type in self.available_models:
            model_func = self.available_models[model_type]
            model = model_func(**model_params)
        else:
            self.logger.error(f"Unknown model type: {model_type}. Using decision_tree instead.")
            model = self.get_decision_tree(**model_params)

        cv_scores = cross_val_score(model, X, y, cv=cv)
        self.logger.info(f"Cross-validation scores: {cv_scores}")
        self.logger.info(f"Mean CV accuracy: {np.mean(cv_scores) * 100:.2f}%")
        self.logger.info(f"CV accuracy std dev: {np.std(cv_scores) * 100:.2f}%")
        return cv_scores

    def hyper_parameter_tuning(self, X, y, model_type='decision_tree', param_grid=None, cv=5):
        """Perform hyperparameter tuning using GridSearchCV"""
        self.logger.info(f"Performing hyperparameter tuning for {model_type}")

        # Default parameter grids if none provided
        default_param_grids = {
            'decision_tree': {
                'max_depth': [3, 5, 7, 10],
                'min_samples_split': [2, 5, 10],
                'criterion': ['gini', 'entropy']
            },
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 10],
                'min_samples_split': [2, 5, 10]
            },
            'svm': {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': ['scale', 'auto', 0.1]
            },
            'knn': {
                'n_neighbors': [3, 5, 7, 11],
                'weights': ['uniform', 'distance'],
                'p': [1, 2]  # 1 for manhattan, 2 for euclidean
            }
        }

        # Use default grid if none provided
        if param_grid is None and model_type in default_param_grids:
            param_grid = default_param_grids[model_type]
        elif param_grid is None:
            self.logger.error(f"No default parameter grid for {model_type} and none provided. Using decision_tree.")
            model_type = 'decision_tree'
            param_grid = default_param_grids['decision_tree']

        # Get model function
        if model_type in self.available_models:
            model_func = self.available_models[model_type]
            model = model_func()
        else:
            self.logger.error(f"Unknown model type: {model_type}. Using decision_tree instead.")
            model = self.get_decision_tree()

        # Perform grid search
        grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='accuracy')
        grid_search.fit(X, y)

        self.logger.info(f"Best parameters: {grid_search.best_params_}")
        self.logger.info(f"Best cross-validation accuracy: {grid_search.best_score_ * 100:.2f}%")

        return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_

    def compare_models(self, X_train, X_test, y_train, y_test, models=None):
        """Compare multiple models on the same dataset"""
        self.logger.info("Comparing multiple models")

        # Default to all available models if none specified
        if models is None:
            models = {name: func() for name, func in self.available_models.items()}

        results = {}
        for name, model in models.items():
            self.logger.info(f"Training and evaluating {name}")
            model.fit(X_train, y_train)
            train_acc = accuracy_score(y_train, model.predict(X_train))
            test_acc = accuracy_score(y_test, model.predict(X_test))
            results[name] = {
                'model': model,
                'train_accuracy': train_acc * 100,
                'test_accuracy': test_acc * 100
            }
            self.logger.info(f"{name} - Train accuracy: {train_acc * 100:.2f}%, Test accuracy: {test_acc * 100:.2f}%")

        # Find the best model based on test accuracy
        best_model_name = max(results, key=lambda x: results[x]['test_accuracy'])
        best_model = results[best_model_name]['model']
        best_acc = results[best_model_name]['test_accuracy']
        self.logger.info(f"Best model: {best_model_name} with test accuracy: {best_acc:.2f}%")

        return results, best_model_name, best_model
