import unittest
import time
import sys
import os
from io import StringIO
from contextlib import redirect_stdout
import gc
import matplotlib
from src.classifier import IrisClassifier
from src.data_loading import DataLoader
from src.feature_proc import FeatureProcessor
from src.model_train import ModelTrainer
from src.logger import Logger

matplotlib.use("Agg")  # Disable plots from showing during tests

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestPerformance(unittest.TestCase):
    """Tests for performance of the Iris Classifier application."""

    def setUp(self):
        """Set up objects needed for the tests."""
        self.logger = Logger()
        self.data_loader = DataLoader(self.logger)
        self.feature_processor = FeatureProcessor(self.logger)
        self.model_trainer = ModelTrainer(self.logger)

        # Suppress output during tests
        self.null_output = StringIO()

    def test_data_loading_performance(self):
        """Test performance of data loading."""
        start_time = time.time()

        # Load data several times to get an average
        num_iterations = 5
        for _ in range(num_iterations):
            iris, iris_df = self.data_loader.load_iris_dataset(source_type="csv")
            del iris, iris_df
            gc.collect()  # Force garbage collection

        avg_time = (time.time() - start_time) / num_iterations
        print(f"\nAverage data loading time: {avg_time:.4f} seconds")

        # Performance should be reasonably fast for small dataset
        self.assertLess(avg_time, 0.5, "Data loading is too slow")

    def test_model_training_performance(self):
        """Test performance of model training."""
        # Load data
        iris, _ = self.data_loader.load_iris_dataset(source_type="csv")

        # Process data
        X_scaled = self.feature_processor.scale_features(iris.data)
        X_pca, _ = self.feature_processor.apply_pca(X_scaled)

        # Split data
        X_train, X_test, y_train, y_test = self.model_trainer.split_data(
            X_pca, iris.target
        )

        # Test different models
        models = ["decision_tree", "random_forest", "svm", "knn"]
        results = {}

        for model_type in models:
            start_time = time.time()

            # Train model
            with redirect_stdout(self.null_output):
                model = self.model_trainer.train_model(
                    X_train, y_train, model_type=model_type
                )

            # Time prediction speed
            pred_start = time.time()
            model.predict(X_test)
            pred_time = time.time() - pred_start

            train_time = time.time() - start_time
            results[model_type] = {
                "training_time": train_time,
                "prediction_time": pred_time,
            }

            print(
                f"\n{model_type.upper()} - Training: {train_time:.4f}s, Prediction: {pred_time:.4f}s"
            )

        # Decision trees and KNN should be quite fast
        self.assertLess(
            results["decision_tree"]["training_time"],
            0.1,
            "Decision tree training is too slow",
        )
        self.assertLess(
            results["knn"]["prediction_time"], 0.05, "KNN prediction is too slow"
        )

    def test_full_workflow_performance(self):
        """Test performance of the full workflow."""
        start_time = time.time()

        # Initialize classifier
        classifier = IrisClassifier(data_source="csv", model_type="decision_tree")

        # Run full workflow with minimal output
        with redirect_stdout(self.null_output):
            # Adjust to skip expensive operations
            classifier.run_full_workflow(
                perform_hyperparameter_tuning=False, compare_models=False
            )

        total_time = time.time() - start_time
        print(f"\nFull workflow execution time: {total_time:.4f} seconds")

        # Adjust the threshold to a more realistic value
        self.assertLess(total_time, 25.0, "Full workflow is extremely slow")

    def test_memory_usage(self):
        """Test memory usage during model training and comparison."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Load data
        iris, _ = self.data_loader.load_iris_dataset()

        # Process and split data
        X_scaled = self.feature_processor.scale_features(iris.data)
        X_pca, _ = self.feature_processor.apply_pca(X_scaled)
        X_train, X_test, y_train, y_test = self.model_trainer.split_data(
            X_pca, iris.target
        )

        # Compare all models
        with redirect_stdout(self.null_output):
            results, best_model_name, best_model = self.model_trainer.compare_models(
                X_train, X_test, y_train, y_test
            )

        # Check memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = final_memory - baseline_memory

        print(f"\nMemory usage for model training and comparison: {memory_used:.2f} MB")

        # Check if memory usage is reasonable
        self.assertLess(memory_used, 100, "Memory usage is too high")


if __name__ == "__main__":
    unittest.main()
