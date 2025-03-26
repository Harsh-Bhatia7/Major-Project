import numpy as np
from .logger import Logger
from .data_loading import DataLoader
from .feature_proc import FeatureProcessor
from .model_train import ModelTrainer
from .model_eval import ModelEvaluator
from .model_storage import ModelSaver


class IrisClassifier:
    """Main class to handle the iris classification workflow"""

    def __init__(self):
        """Initialize the iris classifier with all necessary components"""
        self.logger = Logger()
        self.data_loader = DataLoader(self.logger)
        self.feature_processor = FeatureProcessor(self.logger)
        self.model_trainer = ModelTrainer(self.logger)
        self.model_evaluator = ModelEvaluator(self.logger)
        self.model_saver = ModelSaver(self.logger)

        # Store model artifacts
        self.iris = None
        self.model = None
        self.scaler = None
        self.pca = None

    def run_full_workflow(self):
        """Run the complete iris classification workflow"""
        self.logger.info("Step 1: Project definition - Iris flower classification using Decision Tree")

        # Load data
        self.iris, iris_df = self.data_loader.load_iris_dataset()
        self.data_loader.check_missing_values()

        # Feature processing
        X_scaled = self.feature_processor.scale_features(self.iris.data)
        X_pca, pca_feature_names = self.feature_processor.apply_pca(X_scaled)
        self.feature_processor.visualize_pca(X_pca, self.iris.target)

        # Store preprocessors for future use
        self.scaler = self.feature_processor.scaler
        self.pca = self.feature_processor.pca

        # Model training and evaluation
        cv_scores = self.model_trainer.perform_cross_validation(X_pca, self.iris.target)
        X_train, X_test, y_train, y_test = self.model_trainer.split_data(X_pca, self.iris.target)
        self.model = self.model_trainer.train_decision_tree(X_train, y_train)

        accuracy, conf_matrix, class_report, y_pred = self.model_evaluator.evaluate_model(
            self.model, X_test, y_test, self.iris.target_names
        )

        self.model_evaluator.plot_confusion_matrix(y_test, y_pred, self.iris.target_names)
        self.model_evaluator.visualize_decision_tree(self.model, pca_feature_names, self.iris.target_names)

        # Save the model
        self.model_saver.create_model_directory()
        self.model_saver.save_model_and_preprocessors(self.model, self.scaler, self.pca)

        # Test the model with example prediction
        self.test_prediction(X_test, y_test)

        self.logger.info("Iris Decision Tree Classification workflow completed successfully")

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
        sample = X_test[sample_idx]
        true_species = self.iris.target_names[y_test[sample_idx]]

        # Get original features for this sample
        original_idx = np.where((self.iris.target == y_test[sample_idx]))[0][0]
        original_features = self.iris.data[original_idx]

        sepal_length, sepal_width, petal_length, petal_width = original_features
        predicted_species = self.predict_iris_species(sepal_length, sepal_width, petal_length, petal_width)

        self.logger.info(f"Sample features: Sepal Length: {sepal_length}, Sepal Width: {sepal_width}, Petal Length: {petal_length}, Petal Width: {petal_width}")
        self.logger.info(f"True species: {true_species}")
        self.logger.info(f"Predicted species: {predicted_species}")
