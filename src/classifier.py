import numpy as np
from .logger import Logger
from .data_loading import DataLoader
from .feature_proc import FeatureProcessor
from .model_train import ModelTrainer
from .model_eval import ModelEvaluator
from .model_storage import ModelSaver


class IrisClassifier:
    """Main class to handle the iris classification workflow"""

    def __init__(self, data_source="db", model_type="decision_tree"):
        """
        Initialize the iris classifier with all necessary components

        Args:
            data_source (str): 'db' for database or 'csv' for CSV file
            model_type (str): Type of model to use ('decision_tree', 'random_forest', 'svm', 'knn')
        """
        self.logger = Logger()
        self.data_loader = DataLoader(self.logger)
        self.feature_processor = FeatureProcessor(self.logger)
        self.model_trainer = ModelTrainer(self.logger)
        self.model_evaluator = ModelEvaluator(self.logger)
        self.model_saver = ModelSaver(self.logger)
        self.data_source = data_source
        self.model_type = model_type

        # Store model artifacts
        self.iris = None
        self.model = None
        self.scaler = None
        self.pca = None

    def run_full_workflow(self, perform_hyperparameter_tuning=True, compare_models=True):
        """
        Run the complete iris classification workflow

        Args:
            perform_hyperparameter_tuning (bool): Whether to perform hyperparameter tuning
            compare_models (bool): Whether to compare different models
        """
        self.logger.info(f"Step 1: Project definition - Iris flower classification using {self.model_type}")

        # Load data from database by default
        self.iris, iris_df = self.data_loader.load_iris_dataset(source_type=self.data_source)
        self.data_loader.check_missing_values()

        # Feature processing
        X_scaled = self.feature_processor.scale_features(self.iris.data)
        X_pca, pca_feature_names = self.feature_processor.apply_pca(X_scaled)
        self.feature_processor.visualize_pca(X_pca, self.iris.target)

        # Store preprocessors for future use
        self.scaler = self.feature_processor.scaler
        self.pca = self.feature_processor.pca

        # Split data
        X_train, X_test, y_train, y_test = self.model_trainer.split_data(X_pca, self.iris.target)

        # Model comparison (if requested)
        if compare_models:
            self.logger.info("Comparing different model types")
            results, best_model_name, best_model = self.model_trainer.compare_models(
                X_train, X_test, y_train, y_test
            )
            self.model_type = best_model_name
            self.model = best_model
            self.logger.info(f"Selected {best_model_name} as the best model based on test accuracy")

        # Hyperparameter tuning (if requested)
        elif perform_hyperparameter_tuning:
            self.logger.info(f"Performing hyperparameter tuning for {self.model_type}")
            best_model, best_params, _ = self.model_trainer.hyper_parameter_tuning(
                X_train, y_train, model_type=self.model_type
            )
            self.model = best_model
            self.logger.info(f"Using tuned {self.model_type} model with parameters: {best_params}")

        # Standard training
        else:
            # Cross-validation
            _ = self.model_trainer.perform_cross_validation(X_pca, self.iris.target, model_type=self.model_type)
            # Train the model
            self.model = self.model_trainer.train_model(X_train, y_train, model_type=self.model_type)

        # Evaluate model
        accuracy, conf_matrix, class_report, y_pred = self.model_evaluator.evaluate_model(
            self.model, X_test, y_test, self.iris.target_names
        )

        # Visualizations
        self.model_evaluator.plot_confusion_matrix(y_test, y_pred, self.iris.target_names)

        # Decision Tree visualization only works for Decision Tree models
        if self.model_type == 'decision_tree':
            self.model_evaluator.visualize_decision_tree(self.model, pca_feature_names, self.iris.target_names)

        # Save the model
        self.model_saver.create_model_directory()
        self.model_saver.save_model_and_preprocessors(self.model, self.scaler, self.pca)

        # Test the model with example prediction
        self.test_prediction(X_test, y_test)

        self.logger.info(f"Iris {self.model_type} Classification workflow completed successfully")

    def predict_iris_species(self, sepal_length, sepal_width, petal_length, petal_width):
        """Predict iris species from raw measurements"""
        # Create a feature array from input
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

        # Scale the features
        features_scaled = self.scaler.transform(features)

        # Apply PCA
        features_pca = self.pca.transform(features_scaled)

        # Make prediction
        species_idx = self.model.predict(features_pca)[0]
        species_name = self.iris.target_names[species_idx]

        return species_name

    def test_prediction(self, X_test, y_test):
        """Test the model with an example"""
        self.logger.info("Example prediction with the saved model:")

        # Sample from the test data
        sample_idx = 0
        true_species = self.iris.target_names[y_test[sample_idx]]

        # Get original features for this sample
        original_idx = np.where((self.iris.target == y_test[sample_idx]))[0][0]
        original_features = self.iris.data[original_idx]

        sepal_length, sepal_width, petal_length, petal_width = original_features
        predicted_species = self.predict_iris_species(sepal_length, sepal_width, petal_length, petal_width)

        self.logger.info(f"Sample features: Sepal Length: {sepal_length}, Sepal Width: {sepal_width}, Petal Length: {petal_length}, Petal Width: {petal_width}")
        self.logger.info(f"True species: {true_species}")
        self.logger.info(f"Predicted species: {predicted_species}")
