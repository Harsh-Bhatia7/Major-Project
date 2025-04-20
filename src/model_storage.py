import os
import joblib


class ModelSaver:
    """Class to handle model saving and loading"""

    def __init__(self, logger):
        """Initialize with logger"""
        self.logger = logger
        self.model_dir = None

    def create_model_directory(self, dir_name='iris_model'):
        """Create directory to save model artifacts"""
        self.model_dir = os.path.join(os.getcwd(), dir_name)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        return self.model_dir

    def save_model_and_preprocessors(self, model, scaler, pca):
        """Save the trained model and preprocessors"""
        self.logger.info("Step 7: Saving the trained model and preprocessors")

        if not self.model_dir:
            self.create_model_directory()

        # Save the model, scaler, and PCA
        joblib.dump(model, os.path.join(self.model_dir, 'decision_tree_model.pkl'))
        joblib.dump(scaler, os.path.join(self.model_dir, 'scaler.pkl'))
        joblib.dump(pca, os.path.join(self.model_dir, 'pca.pkl'))
        self.logger.info(f"Model and preprocessors saved to {self.model_dir} directory")

    def load_model_and_preprocessors(self):
        """Load the trained model and preprocessors"""
        model_path = os.path.join(self.model_dir, 'decision_tree_model.pkl')
        scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
        pca_path = os.path.join(self.model_dir, 'pca.pkl')

        if not all(os.path.exists(p) for p in [model_path, scaler_path, pca_path]):
            self.logger.warning("Model files not found. You'll need to train a new model.")
            raise FileNotFoundError("Model files not found")

        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        pca = joblib.load(pca_path)
        return model, scaler, pca
