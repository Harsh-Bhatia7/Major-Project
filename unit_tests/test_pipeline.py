import unittest
import sys
import os
from unittest.mock import MagicMock
from io import StringIO
from contextlib import redirect_stdout
import matplotlib
from src.classifier import IrisClassifier

matplotlib.use('Agg')  # Disable plots from showing during tests

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestPipeline(unittest.TestCase):
    """Tests for the full pipeline execution of the classifier."""

    def setUp(self):
        """Set up objects needed for tests."""
        # Create a mock logger to prevent console output during tests
        self.mock_logger = MagicMock()

        # Temporarily redirect stdout to suppress output
        self.null_output = StringIO()

    def test_end_to_end_workflow(self):
        """Test the complete workflow from data loading to prediction."""
        with redirect_stdout(self.null_output):  # Suppress stdout during test
            # Initialize classifier with test settings
            classifier = IrisClassifier(data_source="csv", model_type="decision_tree")

            # Run full workflow
            classifier.run_full_workflow(perform_hyperparameter_tuning=False, compare_models=False)

            # Verify model and preprocessors are created
            self.assertIsNotNone(classifier.model, "Model was not created")
            self.assertIsNotNone(classifier.scaler, "Scaler was not created")
            self.assertIsNotNone(classifier.pca, "PCA was not created")

            # Test prediction functionality with known input for setosa
            species = classifier.predict_iris_species(
                sepal_length=5.1,
                sepal_width=3.5,
                petal_length=1.4,
                petal_width=0.2
            )

            # Setosa has these distinctive measurements
            self.assertEqual(species, "Iris-setosa",
                            f"Prediction failed: expected Iris-setosa, got {species}")

    def test_model_switching_pipeline(self):
        """Test switching between different model types in the pipeline."""
        with redirect_stdout(self.null_output):
            # Test with all available model types
            model_types = ['decision_tree', 'random_forest', 'svm', 'knn']

            for model_type in model_types:
                # Initialize classifier with the current model type
                classifier = IrisClassifier(data_source="csv", model_type=model_type)

                # Run workflow without hyperparameter tuning to save time
                classifier.run_full_workflow(perform_hyperparameter_tuning=False, compare_models=False)

                # Verify model type matches what we requested
                self.assertEqual(classifier.model_type, model_type,
                               f"Model type mismatch: expected {model_type}")

                # Verify model can make predictions
                species = classifier.predict_iris_species(6.3, 3.3, 6.0, 2.5)  # Typical virginica
                self.assertIn(species, classifier.iris.target_names,
                            f"Invalid species prediction: {species}")

    def test_data_processing_pipeline(self):
        """Test the data processing pipeline with different data sources."""
        with redirect_stdout(self.null_output):
            # Test both data source types
            sources = ["csv", "db"]

            for source in sources:
                classifier = IrisClassifier(data_source=source)

                # Only load and process data
                classifier.iris, iris_df = classifier.data_loader.load_iris_dataset(source_type=source)
                X_scaled = classifier.feature_processor.scale_features(classifier.iris.data)
                X_pca, _ = classifier.feature_processor.apply_pca(X_scaled)

                # Verify data was processed correctly
                self.assertEqual(X_pca.shape[0], len(classifier.iris.target),
                               f"PCA output rows don't match input rows using {source} source")
                self.assertEqual(X_pca.shape[1], 2,
                               f"PCA should reduce to 2 dimensions, got {X_pca.shape[1]} using {source} source")

    def test_model_comparison_pipeline(self):
        """Test the model comparison functionality in the pipeline."""
        with redirect_stdout(self.null_output):
            # Initialize classifier
            classifier = IrisClassifier()

            # Load and process data
            classifier.iris, _ = classifier.data_loader.load_iris_dataset(source_type="csv")
            X_scaled = classifier.feature_processor.scale_features(classifier.iris.data)
            X_pca, _ = classifier.feature_processor.apply_pca(X_scaled)

            # Split data
            X_train, X_test, y_train, y_test = classifier.model_trainer.split_data(
                X_pca, classifier.iris.target
            )

            # Compare models
            results, best_model_name, best_model = classifier.model_trainer.compare_models(
                X_train, X_test, y_train, y_test
            )

            # Verify comparison results
            self.assertGreaterEqual(len(results), 4, "Should compare at least 4 models")
            for model_name, result in results.items():
                self.assertIn('train_accuracy', result, f"{model_name} missing train_accuracy")
                self.assertIn('test_accuracy', result, f"{model_name} missing test_accuracy")
                self.assertIn('model', result, f"{model_name} missing model object")

            # Verify best model was selected
            self.assertIsNotNone(best_model_name, "No best model name returned")
            self.assertIsNotNone(best_model, "No best model object returned")

    def test_training_prediction_pipeline(self):
        """Test the pipeline from training through prediction."""
        with redirect_stdout(self.null_output):
            # Initialize with a specific model type
            classifier = IrisClassifier(model_type="random_forest")

            # Load data
            classifier.iris, _ = classifier.data_loader.load_iris_dataset()

            # Process features
            X_scaled = classifier.feature_processor.scale_features(classifier.iris.data)
            X_pca, _ = classifier.feature_processor.apply_pca(X_scaled)
            classifier.scaler = classifier.feature_processor.scaler
            classifier.pca = classifier.feature_processor.pca

            # Train model directly
            X_train, X_test, y_train, y_test = classifier.model_trainer.split_data(X_pca, classifier.iris.target)
            classifier.model = classifier.model_trainer.train_model(X_train, y_train, model_type="random_forest")

            # Test prediction on each class to verify pipeline consistency
            test_samples = {
                "Iris-setosa": [5.1, 3.5, 1.4, 0.2],
                "Iris-versicolor": [6.0, 2.2, 4.0, 1.0],
                "Iris-virginica": [6.3, 3.3, 6.0, 2.5]
            }

            # Make predictions and check general accuracy
            correct_count = 0
            for expected_class, measurements in test_samples.items():
                sepal_length, sepal_width, petal_length, petal_width = measurements
                predicted_class = classifier.predict_iris_species(
                    sepal_length, sepal_width, petal_length, petal_width
                )
                if predicted_class == expected_class:
                    correct_count += 1

            # Should get at least 2/3 correct with a decent model
            self.assertGreaterEqual(correct_count, 2,
                                  f"Model only predicted {correct_count}/3 test samples correctly")

    def test_model_persistence_pipeline(self):
        """Test the model saving and loading pipeline."""
        with redirect_stdout(self.null_output):
            # First classifier to create and save a model
            classifier1 = IrisClassifier(model_type="decision_tree")

            # Initialize and train
            classifier1.run_full_workflow(perform_hyperparameter_tuning=False, compare_models=False)

            # Get a prediction to use as reference
            reference_sample = [5.0, 3.6, 1.4, 0.2]  # Known setosa sample
            prediction1 = classifier1.predict_iris_species(*reference_sample)

            # Create a new instance and load the saved model
            classifier2 = IrisClassifier()
            classifier2.model_saver.create_model_directory()
            classifier2.model, classifier2.scaler, classifier2.pca = (
                classifier2.model_saver.load_model_and_preprocessors()
            )

            # Load iris dataset for target names
            classifier2.iris, _ = classifier2.data_loader.load_iris_dataset()

            # Make the same prediction with the loaded model
            prediction2 = classifier2.predict_iris_species(*reference_sample)

            # Verify predictions match, confirming model persistence works
            self.assertEqual(prediction1, prediction2,
                           f"Predictions don't match: {prediction1} vs {prediction2}")


# Update the run_all_tests.py to include the new test class
def update_test_runner():
    test_runner_path = os.path.join(
        os.path.dirname(__file__), 'run_all_tests.py'
    )

    if os.path.exists(test_runner_path):
        with open(test_runner_path, 'r') as f:
            content = f.read()

        # Only update if not already included
        if "TestPipeline" not in content:
            import_line = "from unit_tests.test_pipeline import TestPipeline"
            test_suite_line = "test_suite.addTest(unittest.makeSuite(TestPipeline))"

            # Add import
            content = content.replace(
                "from unit_tests.test_database import TestDatabase",
                "from unit_tests.test_database import TestDatabase\n" + import_line
            )

            # Add test suite
            content = content.replace(
                "test_suite.addTest(unittest.makeSuite(TestDatabase))",
                "test_suite.addTest(unittest.makeSuite(TestDatabase))\n    " + test_suite_line
            )

            # Write updated content
            with open(test_runner_path, 'w') as f:
                f.write(content)

            print("Updated run_all_tests.py to include pipeline tests")


if __name__ == '__main__':
    update_test_runner()
    unittest.main()
